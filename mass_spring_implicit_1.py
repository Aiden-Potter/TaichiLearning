import taichi as ti

# ti.init(debug=True)
ti.init(arch=ti.gpu)

max_num_particles = 256

dt = 1e-3

num_particles = ti.var(ti.i32, shape=())
spring_stiffness = ti.var(ti.f32, shape=())
paused = ti.var(ti.i32, shape=())
damping = ti.var(ti.f32, shape=())

particle_mass = 1
bottom_y = 0.05
top_y = 0.95
left_x = 0.05
right_x = 0.95

x = ti.Vector(2, dt=ti.f32, shape=max_num_particles)
v = ti.Vector(2, dt=ti.f32, shape=max_num_particles)

A = ti.Matrix(2, 2, dt=ti.f32, shape=(max_num_particles, max_num_particles))
b = ti.Vector(2, dt=ti.f32, shape=max_num_particles)
r = ti.Vector(2, dt=ti.f32, shape=())
dv = ti.Vector(2, dt=ti.f32, shape=max_num_particles)
new_dv = ti.Vector(2, dt=ti.f32, shape=max_num_particles)


@ti.func
def iterate():
    n = num_particles[None]
    for i in range(n):
        r[None].x = b[i].x
        r[None].y = b[i].y

        for j in range(n):
            if i != j:
                r[None] -= A[i, j] @ dv[j]

        # new_dv[i] = A[i, i].inverse() @ r
        new_dv[i].x = (r[None].x - A[i, i][0, 1] * dv[i].y) / A[i, i][0, 0]
        new_dv[i].y = (r[None].y - A[i, i][1, 0] * dv[i].x) / A[i, i][1, 1]

    for i in range(n):
        dv[i] = new_dv[i]
        # dv[i].x = new_dv[i].x
        # dv[i].y = new_dv[i].y


@ti.kernel
def residual() -> ti.f32:
    n = num_particles[None]
    res = 0.0

    for i in range(n):
        r = b[i] * 1.0
        for j in range(n):
            r -= A[i, j] @ dv[j]
        res += r.x ** 2 + r.y ** 2

    return res


@ti.func
def resi() -> ti.f32:
    n = num_particles[None]
    res = 0.0

    for i in range(n):
        r = b[i] * 1.0
        for j in range(n):
            r -= A[i, j] @ dv[j]
        res += r.x ** 2 + r.y ** 2

    return res


# rest_length[i, j] = 0 means i and j are not connected
rest_length = ti.var(ti.f32, shape=(max_num_particles, max_num_particles))

connection_radius = 0.15

gravity = [0, -9.8]


@ti.func
def solve_equation():
    for i in range(1000):
        iterate()
        if resi() < 1e-9:
            break
        # print(f'iter {i}, residual={residual():0.10f}')


@ti.kernel
def substep():
    # Compute force and new velocity
    n = num_particles[None]
    for i in range(n):
        v[i] *= ti.exp(-dt * damping[None])  # damping
        total_force = ti.Vector(gravity) * particle_mass
        for j in range(n):
            if rest_length[i, j] != 0:
                x_ij = x[i] - x[j]
                total_force += -spring_stiffness[None] * (x_ij.norm() - rest_length[i, j]) * x_ij.normalized()
        v[i] += dt * total_force / particle_mass

    # Collide with ground
    for i in range(n):
        if x[i].y < bottom_y:
            x[i].y = bottom_y
            v[i].y = 0

    # Compute new position
    for i in range(num_particles[None]):
        x[i] += v[i] * dt

    df_ij(0, 0)
    f_ij(0, 0)


@ti.func
def f_ij(i, j):
    x_ij = x[j] - x[i]  # from i to j
    x_d = x_ij.normalized()
    x_n = x_ij.norm()
    return spring_stiffness[None] * (x_n - rest_length[i, j]) * x_d # 返回弹力的大小


@ti.func
def df_ij(i, j):
    x_ij = x[j] - x[i]  # from i to j
    x_d = x_ij.normalized()
    x_n = x_ij.norm()
    x_o = x_d.outer_product(x_d) # 张量积
    x_oi = x_o - [[1, 0], [0, 1]]
    # print("x_o:",i,j,x_o)
    # print("x_oi",i,j,x_oi)
    return - spring_stiffness[None] * ((x_n - rest_length[i, j]) / x_n * x_oi - x_o)


@ti.kernel
def substep_jacobi():
    # fill A b
    n = num_particles[None]
    dt2 = dt * dt

    for i in range(n):
        for j in range(n):
            if i == j:
                A[i, j] = [[1, 0], [0, 1]]
            elif rest_length[i, j] != 0:
                A[i, j] = - dt2 * df_ij(i, j) # 求力的导数
            else:
                A[i, j] = [[0, 0], [0, 0]]
            # print ("A[i,j]:",A[i,j])
    for i in range(n):
        b[i] = ti.Vector(gravity) * dt
        for j in range(n):
            if i != j and rest_length[i, j] != 0:

                # print("f_ij(i, j):",i,j,f_ij(i, j))
                # print("dt2 *df_ij(i, j):",i,j,dt2 *df_ij(i, j))
                # print("v[j]:",j,v[j])
                # print("dt2 * df_ij(i, j) @ v[j]",i,j,dt2 * df_ij(i, j) @ v[j])# 这个符号是变二维为一维，矩阵的每行和向量进行点乘，称为矩阵乘法
                b[i] += dt * f_ij(i, j) + dt2 * df_ij(i, j) @ v[j]
                # print("b[i]",i,b[i])

    # solve delta_v
    solve_equation()

    # update v
    for i in range(n):
        v[i] *= ti.exp(-dt * damping[None])  # damping
        v[i] += dv[i]

    # Collide with ground
    for i in range(n):
        if x[i].y < bottom_y:
            x[i].y = bottom_y
            v[i].y = 0
        if x[i].y > top_y:
            x[i].y = top_y
            v[i].y = 0

    # Collide with wall
    for i in range(n):
        if x[i].x < left_x:
            x[i].x = left_x
            v[i].x = 0
        if x[i].x > right_x:
            x[i].x = right_x
            v[i].x = 0

    # Compute new position
    for i in range(num_particles[None]):
        x[i] += v[i] * dt


@ti.kernel
def new_particle(pos_x: ti.f32, pos_y: ti.f32):  # Taichi doesn't support using Matrices as kernel arguments yet
    new_particle_id = num_particles[None]
    x[new_particle_id] = [pos_x, pos_y]
    v[new_particle_id] = [0, 0]
    num_particles[None] += 1

    # Connect with existing particles
    for i in range(new_particle_id):
        dist = (x[new_particle_id] - x[i]).norm()
        if dist < connection_radius:
            rest_length[i, new_particle_id] = 0.1
            rest_length[new_particle_id, i] = 0.1


gui = ti.GUI('Mass Spring System', res=(512, 512), background_color=0xdddddd)

spring_stiffness[None] = 10000
damping[None] = 20

new_particle(0.3, 0.3)
new_particle(0.3, 0.4)
new_particle(0.4, 0.4)

while True:
    for e in gui.get_events(ti.GUI.PRESS):
        if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
            exit()
        elif e.key == gui.SPACE:
            paused[None] = not paused[None]
        elif e.key == ti.GUI.LMB:
            new_particle(e.pos[0], e.pos[1])
        elif e.key == 'c':
            num_particles[None] = 0
            rest_length.fill(0)
        elif e.key == 's':
            if gui.is_pressed('Shift'):
                spring_stiffness[None] /= 1.1
            else:
                spring_stiffness[None] *= 1.1
        elif e.key == 'd':
            if gui.is_pressed('Shift'):
                damping[None] /= 1.1
            else:
                damping[None] *= 1.1

    if not paused[None]:
        for step in range(10):
            # substep()
            substep_jacobi()
            # print(f'residual={residual():0.10f}')

    X = x.to_numpy()
    gui.circles(X[:num_particles[None]], color=0xffaa77, radius=5)

    gui.line(begin=(0.0, bottom_y), end=(1.0, bottom_y), color=0x0, radius=1)

    for i in range(num_particles[None]):
        for j in range(i + 1, num_particles[None]):
            if rest_length[i, j] != 0:
                gui.line(begin=X[i], end=X[j], radius=2, color=0x445566)
    gui.text(content=f'C: clear all; Space: pause', pos=(0, 0.95), color=0x0)
    gui.text(content=f'S: Spring stiffness {spring_stiffness[None]:.1f}', pos=(0, 0.9), color=0x0)
    gui.text(content=f'D: damping {damping[None]:.2f}', pos=(0, 0.85), color=0x0)
    gui.show()