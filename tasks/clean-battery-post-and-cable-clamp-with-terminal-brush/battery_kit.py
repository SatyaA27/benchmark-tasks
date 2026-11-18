"""Reusable battery-terminal brush: procedural cup, sprung internal bristle sectors and cone.

Models Osborn 0008493100, 120 mm overall with 65 mm steel bristle trim.
Estimates: body diameter 35 mm, 55 mm body, internal bristle profile 25 mm
mouth tapering to 15.2 mm, four 5 kN/m radial carriers (3 mm travel, 3 g each),
0.7/0.6 friction, 90 g total. Bristle carriers replace many flexible wires;
force is inferred from their physical spring deflection, never from proximity.
All dimensions metres. No imported meshes. Cup opens toward local -z.
"""
import math
from dataclasses import replace
import numpy as np
from yam_parts import box,cyl
from yam_joints import Link,Slide
from yam_mesh import Mesh,MeshMechanism,MeshPart,revolve as _revolve

def revolve(name,profile,segments=32,**kw):
    return _revolve(profile,samples=segments,name=name,**kw)

BRISTLE_K=5000.
BRISTLE_NAMES=('east','north','west','south')
BODY_H=.055
GRIP_Z=.055

def sector(name,profile,a0,a1,color=(.33,.34,.36)):
    # Closed radial cross-section swept through a convex angular sector.
    n=len(profile); angles=np.linspace(a0,a1,4)
    p=np.array([(r*math.cos(a),r*math.sin(a),z) for a in angles for r,z in profile]);f=[]
    for i in range(len(angles)-1):
        for j in range(n):
            a=i*n+j;b=i*n+(j+1)%n;c=a+n;d=b+n
            f.extend(((a,b,d),(a,d,c)))
    f.extend((0,j+1,j) for j in range(1,n-1));last=(len(angles)-1)*n
    f.extend((last,last+j,last+j+1) for j in range(1,n-1))
    f=np.array(f);t=(p-p.mean(0))[f]
    if np.einsum('ij,ij->i',t[:,0],np.cross(t[:,1],t[:,2])).sum()<0:f=f[:,::-1]
    lo,hi=p.min(0),p.max(0)
    return Mesh(name,p,f,collision='convex',color=color,proxy=(box((lo+hi)/2,hi-lo),))

def ring(name,r0,r1,z0,z1,color=(.12,.13,.14),n=24):
    return tuple(sector(name+str(i),((r0,z0),(r1,z0),(r1,z1),(r0,z1)),i*2*math.pi/n,(i+1)*2*math.pi/n,color) for i in range(n))

def brush(pos):
    meshes={'base': ring('shell',.014,.0175,0,.022)+(
        revolve('body',[(0,.022),(.0175,.022),(.0175,.055),(0,.055)],segments=32,collision='convex',color=(.11,.12,.13)),
        revolve('core',[(0,.055),(.0015,.055),(.0015,.120),(0,.120)],segments=12,collision='convex',color=(.45,.46,.48),proxy=(cyl((0,0,.075),.0015,.040),)),
    )}
    links=[]
    profile=((.0125,0),(.0134,0),(.0134,.020),(.0076,.020),(.0086,.001))
    for i,name in enumerate(BRISTLE_NAMES):
        a=i*math.pi/2;axis='x' if i%2==0 else 'y';sign=1 if i<2 else -1
        anchor=(.011*math.cos(a),.011*math.sin(a),.01)
        links.append(Link(name,(),Slide(axis,anchor,-.003 if sign<0 else 0,.003 if sign>0 else 0,spring=BRISTLE_K,damping=3.,armature=.003),.003,friction=(.7,.6)))
        meshes[name]=tuple(sector(name+str(j),profile,a+math.radians(-43+j*29),a+math.radians(-43+(j+1)*29),(.36,.37,.38)) for j in range(3))
    # External cone is a visual bristle envelope in this post-only probe.
    meshes['base']+= (revolve('cone',[(.0015,.055),(.0125,.055),(.0075,.120),(.0015,.120)],segments=24,collision='convex',color=(.32,.33,.35),proxy=(cyl((0,0,.075),.009,.040),)),)
    return MeshMechanism('brush',pos,(),links,mass=.078,color=(.11,.12,.13),friction=(.8,.7),contact_offset=.0005,grip=24,meshes=meshes)

def post(pos):
    return MeshPart('post',pos,(),meshes=(revolve('taper',[(0,0),(.00895,0),(.00795,.018),(0,.018)],segments=40,collision='convex',color=(.65,.66,.68)),),contact_offset=.0005)

def bristle_force(state):
    return BRISTLE_K*sum(abs(float(state.joints[k])) for k in BRISTLE_NAMES)

# The full task replaces the external rigid probe envelope with compliant wires.
_probe_brush=brush
CONE_NAMES=('cone_e','cone_n','cone_w','cone_s')
CONE_K=800.

def brush(pos):
    old=_probe_brush(pos)
    meshes=dict(old.meshes)
    meshes['base']=tuple(m for m in meshes['base'] if m.name!='cone')
    links=list(old.links)
    profile=((.002,.055),(.0125,.055),(.0075,.120),(.002,.120))
    for i,name in enumerate(CONE_NAMES):
        a=i*math.pi/2;axis='x' if i%2==0 else 'y';sign=1 if i<2 else -1
        anchor=(.008*math.cos(a),.008*math.sin(a),.085)
        links.append(Link(name,(),Slide(axis,anchor,-.004 if sign>0 else 0,.004 if sign<0 else 0,spring=CONE_K,damping=1.2,armature=.003),.003,friction=(.65,.5)))
        meshes[name]=tuple(replace(sector(name+str(j),profile,a+math.radians(-43+j*29),a+math.radians(-43+(j+1)*29)),proxy=(box((.008*math.cos(a),.008*math.sin(a),.075),(.004,.004,.040)),) if j==1 else ()) for j in range(3))
    return MeshMechanism('brush',pos,(),links,mass=.066,color=old.color,friction=(.8,.7),contact_offset=.0005,grip=24,meshes=meshes)

def cap(pos):
    """Detached dust cap lying horizontally; no retention mechanism is modelled.

    Estimated 38 mm diameter, 75 mm length, 30 g. Check with ruler/scale.
    """
    pieces=ring('capwall',.0177,.019,0,.073)+(revolve('captop',[(0,.073),(.019,.073),(.019,.075),(0,.075)],collision='convex',color=(.12,.13,.14)),)
    R=np.array(((0.,0,1),(0,1,0),(-1,0,0)))
    meshes=[]
    for m in pieces:
        points=m.points@R.T
        lo,hi=points.min(0),points.max(0)
        meshes.append(replace(m,points=points,proxy=(box((lo+hi)/2,hi-lo),)))
    return MeshPart('cap',pos,(),mass=.030,meshes=tuple(meshes),color=(.12,.13,.14),friction=(.7,.6),contact_offset=.0003)

PATCH_N=24
PATCH_ROWS=3
COATING=(.08,.09,.10)
METAL=(.68,.69,.71)

def painted_post(pos):
    meshes=[revolve('postcore',[(0,0),(.00815,0),(.00715,.018),(0,.018)],collision='convex',color=METAL,proxy=(cyl((0,0,.009),.00895,.018),))]
    for k in range(3):
        z0,z1=k*.006,(k+1)*.006;r0=.00895-z0/18;r1=.00895-z1/18
        for j in range(PATCH_N):
            a0=j*2*math.pi/PATCH_N;a1=(j+1)*2*math.pi/PATCH_N
            meshes.append(replace(sector(f'postpatch_{k*PATCH_N+j}',((r0-.0008,z0),(r0,z0),(r1,z1),(r1-.0008,z1)),a0,a1,COATING),proxy=()))
    return MeshPart('post',pos,(),meshes=tuple(meshes),contact_offset=.0005)


def cable_and_clamp(pos):
    """Deka 04154 304.8 mm, 4 AWG cable; 9.65 mm OD and terminal dimensions estimated.

    Three 64.9 mm rods and a 110 mm terminal rod have bending/torsion hinges. Estimated stiffness
    0.12 Nm/rad bending, 0.04 Nm/rad torsion; damping .015. Fixed ring lug at
    the remote end models the real bolted ground attachment. 3 g helper links
    are a numerical mass approximation, not extra real hardware.
    """
    from yam_joints import Hinge
    links=[];meshes={};last=None
    for i in range(4):
        cuts=(0.,.1948/3,2*.1948/3,.1948,.3048)
        x0,x1=cuts[i:i+2]
        for axis in ('y','z','x'):
            name=f'cable_{i}_{axis}'
            wire_end=x1-.025 if i==3 else x1
            shapes=(cyl(((x0+wire_end)/2,0,.010),.004825,wire_end-x0,axis='x'),) if axis=='x' else (cyl((x0,0,.010),.001,.002),)
            links.append(Link(name,shapes,Hinge(axis,(x0,0,.010),-65,65,spring=.04 if axis=='x' else .12,damping=.015,armature=.00002),.035 if axis=='x' else .003,parent=last,color=(.10,.11,.12),friction=(.8,.7)))
            last=name
    end=.3048
    # The terminal is cast around the cable's last rigid segment. Bore vertical.
    for k in range(3):
        z0,z1=k*.006,(k+1)*.006;r0=.00895-z0/18;r1=.00895-z1/18
        for j in range(PATCH_N):
            a0=j*2*math.pi/PATCH_N;a1=(j+1)*2*math.pi/PATCH_N
            if j==0:continue  # 15 degree slit where the bolt closes the clamp
            m=sector(f'clamppatch_{k*PATCH_N+j}',((r0,z0),(r0+.004,z0),(r1+.004,z1),(r1,z1)),a0,a1,COATING)
            p=m.points+np.array([end,0,0]);meshes.setdefault(last,[]).append(Mesh(m.name,p,m.faces,collision='convex',color=m.color,proxy=((box((end+.011,0,.009),(.004,.008,.018)),box((end-.011,0,.009),(.004,.008,.018)),box((end,.011,.009),(.008,.004,.018)),box((end,-.011,.009),(.008,.004,.018))) if k==0 and j==1 else ())))
    clamp_shapes=(box((end-.022,0,.009),(.018,.018,.018)),box((end+.010,.010,.010),(.012,.014,.020)),cyl((end+.010,.010,.0215),.005,.003))
    old=links[-1]
    links[-1]=Link(old.name,old.shapes+clamp_shapes,old.joint,old.mass+.070,parent=old.parent,color=old.color,friction=old.friction)
    return MeshMechanism('cable',pos,(box((0,0,.004),(.020,.020,.008)),),links,mass=None,color=(.11,.12,.13),friction=(.8,.7),contact_offset=.0005,meshes=meshes)


def cable_frame(state):
    """World origin and rotation of the terminal's bore bottom."""
    R,t=state._mechanism.link_transform('cable_3_x',state.joints)
    Rw=state.rotation()@R
    # Last link hinge anchor: x=.1948, z=.010; bore is 110 mm forward.
    o=np.asarray(state.origin)+state.rotation()@t+Rw@np.array([.110,0,-.010])
    return o,Rw
