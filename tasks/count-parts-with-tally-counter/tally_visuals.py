"""Spawn-time markings only. No state hook, visibility updates or physics APIs.

Digits are attached to their physical links in the q=0 frame and rotate solely
with those links. USD/Isaac imports are confined to the spawn path.
"""
import functools
import importlib.abc
import importlib.machinery
import math
import sys
import numpy as np

_TOKEN = object()
DIGITS = ('abcdef','bc','abdeg','abcdg','bcfg','acdfg','acdefg','abc','abcdefg','abcdfg')


def glyph(digit):
    """Permanent seven-segment printing, exact outside dimensions 2.3 x 4 mm."""
    w,h,t=.0023,.004,.00032
    bars=(('a',0,(h-t)/2,w,t),('g',0,0,w,t),('d',0,-(h-t)/2,w,t),
          ('b',(w-t)/2,(h-t)/4,t,(h-t)/2),('c',(w-t)/2,-(h-t)/4,t,(h-t)/2),
          ('e',-(w-t)/2,-(h-t)/4,t,(h-t)/2),('f',-(w-t)/2,(h-t)/4,t,(h-t)/2))
    return [b for b in bars if b[0] in DIGITS[digit]]


def material(stage,path,color,metal=0.,rough=.4,opacity=1.):
    from pxr import Gf,Sdf,UsdShade
    from yam_arena.station import linear_rgb
    mat=UsdShade.Material.Define(stage,path)
    shader=UsdShade.Shader.Define(stage,path+'/Shader')
    shader.CreateIdAttr().Set('UsdPreviewSurface')
    shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*linear_rgb(color)))
    for key,value in (('metallic',metal),('roughness',rough),('opacity',opacity)):
        shader.CreateInput(key,Sdf.ValueTypeNames.Float).Set(value)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
    return mat


def mesh(stage,path,points,faces,mat):
    from pxr import UsdGeom,UsdShade
    geom=UsdGeom.Mesh.Define(stage,path)
    geom.CreatePointsAttr().Set(np.asarray(points,float).tolist())
    geom.CreateFaceVertexCountsAttr().Set([len(f) for f in faces])
    geom.CreateFaceVertexIndicesAttr().Set([i for f in faces for i in f])
    geom.CreateSubdivisionSchemeAttr().Set('none')
    geom.CreateDoubleSidedAttr().Set(True)
    UsdShade.MaterialBindingAPI.Apply(geom.GetPrim()).Bind(mat)
    return geom


def cylinder(stage,path,center,r,h,axis,mat):
    from pxr import Gf,UsdGeom,UsdShade
    g=UsdGeom.Cylinder.Define(stage,path)
    g.CreateRadiusAttr().Set(r);g.CreateHeightAttr().Set(h);g.CreateAxisAttr().Set(axis)
    g.AddTranslateOp().Set(Gf.Vec3d(*map(float,center)))
    UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(mat)


def author(stage,root,spec):
    import tally_counter as c
    from pxr import UsdShade
    base=root+'/base'
    chrome=material(stage,base+'/Chrome',(.70,.73,.77),.95,.16)
    black=material(stage,base+'/ABS',(.018,.02,.023),0.,.48)
    white=material(stage,base+'/Printing',(.97,.97,.94),0.,.6)
    glass=material(stage,base+'/Glass',(.70,.86,.90),0.,.12,.12)
    # Cylindrical shell axis X. Front face has an actual 28 x 7 mm opening.
    n=192;a=np.arange(n)*2*math.pi/n
    yz=np.column_stack((c.CASE_R*np.cos(a),c.WHEEL_Z+c.CASE_R*np.sin(a)))
    points=np.vstack([np.column_stack((np.full(n,x),yz)) for x in (-.0148,.0148)])
    faces=[(i,(i+1)%n,n+(i+1)%n,n+i) for i in range(n)]
    faces.append(tuple(range(n,2*n)))
    mesh(stage,base+'/case_shell',points,faces,chrome)
    half=.0035;r=c.CASE_R
    alpha=math.asin(half/r)
    for label,angles in (('upper',np.linspace(alpha,math.pi-alpha,100)),
                         ('lower',np.linspace(math.pi+alpha,2*math.pi-alpha,100))):
        q=np.column_stack((r*np.cos(angles),c.WHEEL_Z+r*np.sin(angles)))
        mesh(stage,base+'/front_'+label,np.column_stack((np.full(len(q),-.01481),q)),
             [tuple(range(len(q)))],chrome)
    for side in (-1,1):
        # Curved outer sides beside the rectangular window.
        ang=np.linspace(-alpha,alpha,25)
        q=np.column_stack((side*r*np.cos(ang),c.WHEEL_Z+r*np.sin(ang)))
        q=np.vstack((q, (side*.014,c.WHEEL_Z+half),(side*.014,c.WHEEL_Z-half)))
        mesh(stage,base+f'/front_side_{side+1}',np.column_stack((np.full(len(q),-.01482),q)),
             [tuple(range(len(q)))],chrome)
    mesh(stage,base+'/window_glass',[(-.0149,y,c.WHEEL_Z+z) for y,z in
         ((-.014,-half),(.014,-half),(.014,half),(-.014,half))],[(0,1,2,3)],glass)
    for k in range(3):
        phi=(90+120*k)*c.DEG
        cylinder(stage,base+f'/screw_{k}',(c.SCREW_CIRCLE_R*math.cos(phi),
                 c.SCREW_CIRCLE_R*math.sin(phi),.00525),.0025,.0005,'Z',chrome)
    # All 20 moving figures are authored ONCE under their rigid-body prims.
    for name in ('units','tens','hundreds','thousands'):
        moving=name in c.INITIAL_DEG
        parent=root+'/'+name if moving else base
        anchor=np.asarray(spec._link(name).joint.anchor) if moving else np.zeros(3)
        for d in range(10) if moving else (0,):
            qd=(c.INITIAL_DEG[name]+36*(d-c.START_MOVING[name]))*c.DEG if moving else 0.
            for seg,right,up,w,h in glyph(d):
                vertices=[];faces=[]
                # Subdivide the print vertically so the whole 4 mm glyph lies
                # on the actual gently scalloped land, not a floating plane.
                for j in range(9):
                    zz=up-h/2+j*h/8
                    for yy in (right-w/2,right+w/2):
                        # Solve for true projected height on the curved land;
                        # s/R alone would shrink the nominal4 mm glyph.
                        low,high=-12*c.DEG,12*c.DEG
                        for _ in range(35):
                            delta=(low+high)/2
                            radius=(c.LIP_R+(c.WHEEL_R-c.LIP_R)*math.cos(math.pi*delta/(24*c.DEG))) if moving else c.WHEEL_R
                            if radius*math.sin(delta)<zz:low=delta
                            else:high=delta
                        delta=(low+high)/2;phi=math.pi+qd-delta
                        rad=((c.LIP_R+(c.WHEEL_R-c.LIP_R)*math.cos(math.pi*delta/(24*c.DEG))) if moving else c.WHEEL_R)+.000025
                        v=np.array((rad*math.cos(phi),c.WHEEL_YS[name]-yy,
                                    c.WHEEL_Z+rad*math.sin(phi)))-anchor
                        vertices.append(v)
                faces=[(2*j,2*j+1,2*j+3,2*j+2) for j in range(8)]
                mesh(stage,parent+f'/print_{name}_{d}_{seg}',vertices,faces,white)
    # Finely spaced ribs are visual-only; the grip collider is the true13 mm cylinder.
    kp=root+'/knob';anchor=np.asarray(spec._link('knob').joint.anchor)
    for k in range(40):
        a=k*2*math.pi/40
        centre=(.0065*math.cos(a),sum(c.KNOB_Y)/2,.0065*math.sin(a))
        cylinder(stage,kp+f'/knurl_{k}',centre,.00018,c.KNOB_Y[1]-c.KNOB_Y[0]-.0005,'Y',chrome)
    for body in ('base','key','knob'):
        for i,shape in enumerate(spec.base_shapes if body=='base' else spec._link(body).shapes):
            if shape.color==c.CHROME:
                UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(root+f'/{body}/shape_{i}')).Bind(chrome)
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(base+'/shape_0')).Bind(black)


def _patch_scene(scene):
    import yam_mesh
    yam_mesh._patch_scene(scene)
    if getattr(scene,'_tally_visual_token',None) is _TOKEN:return
    inner=scene.task_object
    @functools.wraps(inner)
    def task_object(spec):
        obj=inner(spec)
        if spec.name=='counter' and isinstance(spec,yam_mesh.MeshMechanism):
            from isaaclab.sim.utils import clone
            original=getattr(obj.spawner_cfg.func,'__wrapped__',obj.spawner_cfg.func)
            def spawn(path,cfg,*args,**kwargs):
                prim=original(path,cfg,*args,**kwargs)
                author(prim.GetStage(),str(prim.GetPath()),spec)
                return prim
            obj.spawner_cfg.func=obj.object_cfg.spawn.func=clone(spawn)
        return obj
    scene.task_object=task_object
    scene._tally_visual_token=_TOKEN


class _Finder(importlib.abc.MetaPathFinder):
    _tally_visual_finder=True
    def find_spec(self,fullname,path=None,target=None):
        if fullname!='yam_arena.scene':return None
        import yam_parts
        spec=importlib.machinery.PathFinder.find_spec(fullname,path)
        if spec is not None and spec.loader is not None:
            spec.loader=yam_parts._SceneLoader(spec.loader,_patch_scene)
        return spec


def install():
    if not any(getattr(f,'_tally_visual_finder',False) for f in sys.meta_path):
        sys.meta_path.insert(0,_Finder())
    if 'yam_arena.scene' in sys.modules:_patch_scene(sys.modules['yam_arena.scene'])
