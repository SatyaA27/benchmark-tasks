"""IEC compact-cassette shell, passive six-tooth reels and sharpened hex pencil.
Nominals: shell 100.4 x 63.8 x 12 mm, hub spacing 42.5 mm, tooth/root
circles 7.9/10.2 mm, shell opening 11.1 mm, 7 mm AF pencil. Estimates:
1 mm shell skins, tooth width 1.3 mm, hub depth 4 mm, 24 mm wound diameter,
40 g cassette / 5 g pencil, drag 0.002 Nm. 23-degree sharpener gives a
17.407 mm truncated cone from 1 mm nose to 8.083 mm corner diameter.
All geometry is procedural. Frame: cassette thin axis x, long axis y, height z.
"""
from dataclasses import replace
import math
import numpy as np
from yam_parts import box
from yam_joints import Hinge,Link
from yam_mesh import Mesh,MeshPart,MeshMechanism

AF=.007
R=AF/math.sqrt(3)
CONE=(2*R-.001)/2/math.tan(math.radians(11.5))
HUBS=(-.02125,.02125)
HUB_Z=.004


def loft(name,rings,color):
    p=np.array(rings).reshape(-1,3); n=len(rings[0]); f=[]
    for i in range(len(rings)-1):
        for j in range(n):
            a=i*n+j;b=i*n+(j+1)%n;f.extend([(a,b,b+n),(a,b+n,a+n)])
    f.extend((0,j+1,j) for j in range(1,n-1));a=(len(rings)-1)*n
    f.extend((a,a+j,a+j+1) for j in range(1,n-1));f=np.array(f)
    t=(p-p.mean(0))[f]
    if np.einsum('ij,ij->i',t[:,0],np.cross(t[:,1],t[:,2])).sum()<0:f=f[:,::-1]
    return Mesh(name,p,f,collision='convex',color=color,proxy=(box((p.min(0)+p.max(0))/2,np.maximum(np.ptp(p,axis=0),.0001)),))


def yz_prism(name,poly,x,t,color):
    return loft(name,[[(x-t/2,y,z) for y,z in poly],[(x+t/2,y,z) for y,z in poly]],color)


def sector(name,center,r0,r1,a,b,x,t,color):
    cy,cz=center
    poly=[(cy+r0*math.cos(a),cz+r0*math.sin(a)),(cy+r1*math.cos(a),cz+r1*math.sin(a)),(cy+r1*math.cos(b),cz+r1*math.sin(b)),(cy+r0*math.cos(b),cz+r0*math.sin(b))]
    return yz_prism(name,poly,x,t,color)


def cassette(pos):
    grey=(.52,.60,.64); white=(.88,.87,.80); shell=[]; links=[];meshes={}
    # Each half face extends from the circular 11.1 mm opening to its rectangle.
    for k,cy in enumerate(HUBS):
        ymin,ymax=(-.0502,0) if cy<0 else (0,.0502)
        zmin,zmax=-.0319,.0319
        angles=list(np.linspace(0,2*math.pi,25)[:-1])
        angles+= [math.atan2(z-HUB_Z,y-cy)%(2*math.pi) for y in (ymin,ymax) for z in (zmin,zmax)]
        angles=sorted(set(angles))
        for j,a in enumerate(angles):
            b=angles[(j+1)%len(angles)]+(2*math.pi if j==len(angles)-1 else 0)
            def outer(v):
                dy,dz=math.cos(v),math.sin(v)
                d=min((ymax-cy)/dy if dy>1e-10 else (ymin-cy)/dy if dy< -1e-10 else 1e6,(zmax-HUB_Z)/dz if dz>1e-10 else (zmin-HUB_Z)/dz if dz< -1e-10 else 1e6)
                return (cy+d*dy,HUB_Z+d*dz)
            poly=[(cy+.00555*math.cos(a),HUB_Z+.00555*math.sin(a)),outer(a),outer(b),(cy+.00555*math.cos(b),HUB_Z+.00555*math.sin(b))]
            for side in (-1,1):shell.append(yz_prism(f'skin_{k}_{j}_{side+1}',poly,side*.0055,.001,grey))
        hub=[]
        for j in range(24):hub.append(sector(f'hub_{k}_{j}',(cy,HUB_Z),.0051,.0105,j*math.pi/12,(j+1)*math.pi/12,0,.004,white))
        for j in range(6):
            a=j*math.pi/3;u=np.array([math.cos(a),math.sin(a)]);v=np.array([-u[1],u[0]])
            poly=[np.array([cy,HUB_Z])+r*u+w*v for r,w in ((.00395,-.00065),(.00515,-.00065),(.00515,.00065),(.00395,.00065))]
            hub.append(yz_prism(f'tooth_{k}_{j}',poly,0,.004,white))
        name=f'reel{k}';links.append(Link(name,(),Hinge('x',(0,cy,HUB_Z),-3600,3600,damping=.000002,armature=.00000002,static_friction=.002,dynamic_friction=.0015),.004,color=white));meshes[name]=tuple(hub)
    shapes=[box((0,y,0),(.012,.0012,.0638),grey) for y in (-.0496,.0496)]
    shapes += [box((0,0,z),(.012,.1004,.0012),grey) for z in (-.0313,.0313)]
    meshes['base']=tuple(shell)
    return MeshMechanism('cassette',pos,shapes,links,mass=.032,color=grey,friction=(.5,.4),contact_offset=.0003,grip=1,self_collide=False,meshes=meshes)


def pencil(pos):
    yellow=(.92,.68,.08);wood=(.70,.49,.29)
    angles=np.arange(6)*math.pi/3+math.pi/6
    def ring(x,r):return [(x,r*math.cos(a),r*math.sin(a)) for a in angles]
    barrel=loft('hex_barrel',[ring(-.055,R),ring(.120-CONE,R)],yellow)
    barrel=replace(barrel,proxy=tuple(box(((a+b)/2,0,0),(b-a,.007,2*R)) for a,b in ((-.055,-.015),(-.015,.015),(.015,.120-CONE))))
    cone=loft('sharpened_cone',[ring(.120-CONE,R),ring(.120,.0005)],wood)
    return MeshPart('pencil',pos,(),mass=.005,color=yellow,friction=(.6,.5),contact_offset=.0003,grip=1,meshes=(barrel,cone))
