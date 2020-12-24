import taichi as ti

ti.init(arch = ti.cpu,debug=True)

max_num_particles = 256

dt = 1e-3

num_particles = ti.var(ti.i32, shape=()) # 一个0维的变量，访问和修改需要使用num_particles[None]
spring_stiffness = ti.var(ti.f32, shape=())
paused = ti.var(ti.i32, shape=())
damping = ti.var(ti.f32, shape=())
dashpot_damping = ti.field(dtype=ti.f32, shape=())
particle_mass = 1
bottom_y = 0.05

x = ti.Vector(2, dt=ti.f32, shape=max_num_particles)
#x = ti.Vector.field(2, dt=ti.f32, shape=max_num_particles)
v = ti.Vector(2, dt=ti.f32, shape=max_num_particles)
f = ti.Vector.field(2, dtype=ti.f32, shape=max_num_particles)
A = ti.Matrix(2, 2, dt=ti.f32, shape=(max_num_particles, max_num_particles))
b = ti.Vector(2, dt=ti.f32, shape=max_num_particles)
# 一个数组
fixed = ti.field(dtype=ti.i32,shape=max_num_particles)

# rest_length[i, j] = 0 means i and j are not connected 一个二维数组
rest_length = ti.var(ti.f32, shape=(max_num_particles, max_num_particles))

connection_radius = 0.15
substeps =10
gravity = [0, -9.8]
# 一开始就要声明所有的变量
@ti.kernel
def substep():
    # 将模拟推进一个步长
    # Compute force and new velocity
    n = num_particles[None]
    # Compute force
    for i in range(n):
        # Gravity
        f[i] = ti.Vector([0, -9.8]) * particle_mass
        for j in range(n):
            if rest_length[i, j] != 0:
                x_ij = x[i] - x[j]
                d = x_ij.normalized()
                # Spring force
                f[i] += -spring_stiffness[None] * (x_ij.norm() / rest_length[i, j] -
                                           1) * d
                # # Dashpot damping 相对速度 解决抽搐的问题 减震器，点乘算大小，方向在两点连线上
                # v_rel = (v[i] - v[j]).dot(d)
                # f[i] += -dashpot_damping[None] * v_rel * d
    # Collide with ground
    for i in range(n):
        if x[i].y < bottom_y:
            x[i].y = bottom_y
            v[i].y = 0
        if x[i].x < 0:
            x[i].x = 0
            v[i].x = 0
        if x[i].x > 1:
            x[i].x = 1
            v[i].x = 0
    for i in range(n):
        # 速度的阻尼 在全局作用下的影响
        if not fixed[i]:
            v[i] += dt*f[i] / particle_mass
            v[i] *= ti.exp(-dt * damping[None])  # drag damping
            x[i] += v[i]*dt
        else:
            v[i] = ti.Vector([0,0])
            # fixed则不更新位置



    # # Compute new position
    # # 更新了速度之后计算位移 semi-implicit
    # for i in range(num_particles[None]):
    #     x[i] += v[i] * dt


@ti.kernel
def new_particle(pos_x: ti.f32, pos_y: ti.f32,fixed_:ti.i32): # Taichi doesn't support using Matrices as kernel arguments yet
    new_particle_id = num_particles[None]
    x[new_particle_id] = [pos_x, pos_y]
    v[new_particle_id] = [0, 0]
    num_particles[None] += 1
    fixed[new_particle_id] = fixed_
    
    # Connect with existing particles
    for i in range(new_particle_id):
        dist = (x[new_particle_id] - x[i]).norm()
        if dist < connection_radius:
            rest_length[i, new_particle_id] = 0.1
            rest_length[new_particle_id, i] = 0.1

@ti.kernel
def attract(pos_x:ti.f32,pos_y:ti.f32):
    for i in range(num_particles[None]):
        v[i]+= -dt*substeps*(x[i]-ti.Vector([pos_x,pos_y]))*300
    
gui = ti.GUI('Mass Spring System', res=(512, 512), background_color=0xdddddd)

spring_stiffness[None] = 10000
damping[None] = 20
dashpot_damping[None] = 100
new_particle(0.3, 0.3,0)
new_particle(0.3, 0.4,0)
new_particle(0.4, 0.4,0)

while True:
    for e in gui.get_events(ti.GUI.PRESS):
        if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
            exit()
        elif e.key == gui.SPACE:
            paused[None] = not paused[None]
        elif e.key == ti.GUI.LMB:
            # print(e.pos)
            new_particle(e.pos[0], e.pos[1],gui.is_pressed(ti.GUI.SHIFT))
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
        for step in range(substeps):
            # 每帧跑10次substep
            substep()
    if gui.is_pressed(ti.GUI.RMB):
        c = gui.get_cursor_pos()
        attract(c[0],c[1])
    # 读取taichi的field的粒子位置很慢，转成Numpy去读取
    X = x.to_numpy()
    # 传入一个数组一次性渲染完成
    for i in range(num_particles[None]):
        c = 0xFF0000  if fixed[i] else 0x111111
        gui.circle(X[i], color=c, radius=5)
    
    gui.line(begin=(0.0, bottom_y), end=(1.0, bottom_y), color=0x0, radius=1)
    
    for i in range(num_particles[None]):
        for j in range(i + 1, num_particles[None]):
            if rest_length[i, j] != 0:
                gui.line(begin=X[i], end=X[j], radius=2, color=0x445566)
    gui.text(content=f'C: clear all; Space: pause', pos=(0, 0.95), color=0x0)
    gui.text(content=f'S: Spring stiffness {spring_stiffness[None]:.1f}', pos=(0, 0.9), color=0x0)
    gui.text(content=f'D: damping {damping[None]:.2f}', pos=(0, 0.85), color=0x0)
    gui.show()

