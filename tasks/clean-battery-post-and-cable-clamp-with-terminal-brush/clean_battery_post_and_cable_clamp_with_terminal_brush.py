"""Clean a tapered battery post with the internal end of a terminal brush.

The cleaning rule changes only each patch's colour and remembered coverage;
rigid bodies and spring bristles still enforce contact and load. Coordinates
are metres, x forward, y left, heights relative to the bench in documentation.
"""
import math
import pathlib
import sys
import numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
import importlib
import battery_kit as _battery_kit
importlib.reload(_battery_kit)
from battery_kit import (brush,cap,painted_post,cable_and_clamp,cable_frame,BRISTLE_NAMES,BRISTLE_K,CONE_NAMES,CONE_K,PATCH_N,COATING,METAL)
from yam_parts import Part,box,cyl
from yam_arena.constants import FINGER_OPEN
from yam_arena.demo import Waypoint,grip_rotation
from yam_arena.task_file import TABLE_TOP

SCENE='station'
TOP=TABLE_TOP[SCENE]
SUCCESS_HOLD_S=1.
SCORE_OVER='end'
RUN_LIMIT_STEPS=3200
DIAG=True
DEMO_PRESS_ONLY=False
TEST_COVERAGE=None
TEST_HANDS=None
TEST_HISTORY=None  # offline fixtures only; runtime always records the full trial
_CONTACT_SUB=None
POST_POS=np.array((.240,-.220,TOP+.132))
BRUSH_POS=np.array((.230,-.365,TOP+.001))
CAP_PLACE=np.array((.31,-.34,TOP+.0195))
CABLE_POS=np.array((-.1348,0.,TOP+.001))
POST_HEIGHT=.018
GRIP=.055
PINCH=-.016
CAP_PINCH=-.0175
CABLE_PINCH=-.003825
RETURN_XY=np.array((.23,-.365))

# Runtime clock/state. The detached cap is an ordinary passive loose object.
def _context():
    c={'env':None,'t':None,'episode':0,'step':0}
    try:from yam_arena import inspect_embodiment as ie
    except ImportError:return c
    cls=ie.YamArenaEmbodiment
    if hasattr(cls,'_battery_run'):return cls._battery_run
    oldstep,oldreset=cls._task_step,cls.reset
    def step(self,*a,**kw):
        c['env'],c['t']=self._env,self._t;c['step']+=1
        return oldstep(self,*a,**kw)
    def reset(self,*a,**kw):
        r=oldreset(self,*a,**kw)
        c.update(env=self._env,t=0.,episode=c['episode']+1,step=0)
        _begin_episode()
        _install_contact_reports()
        return r
    cls._task_step,cls.reset=step,reset;cls._battery_run=c
    return c
_RUN=_context()
_MEMORY={}
_READS={}
_PAINT={}

# Scoring first: every nonzero tier requires tool use and no gripper/taper contact.
def _fractions(objects):
    if TEST_COVERAGE is not None:return tuple(float(x) for x in TEST_COVERAGE)
    _update_cleaning(objects)
    if not _MEMORY:return (0.,0.)
    return tuple(float(np.mean(_MEMORY[k]>=.020)) for k in ('post','clamp'))

def _hands():
    if TEST_HANDS is not None:return TEST_HANDS
    if _RUN['env'] is None:return None
    from yam_arena import ik,station
    from yam_arena.constants import ARM_JOINTS,FINGER_JOINTS
    out={}
    for arm in ('left','right'):
        art=_RUN['env'].unwrapped.scene[arm+'_robot']
        ids=art.find_joints(ARM_JOINTS,preserve_order=True)[0]
        q=art.data.joint_pos; q=getattr(q,'torch',q)[0].detach().cpu().numpy()
        out[arm]=ik.tcp_pose(q[ids],station.BASES[arm])[0]
    return out

def returned(objects):
    b=objects['brush'];o=np.asarray(b.origin)
    low=np.asarray(b.low);hands=_hands()
    clear=hands is None or all(np.linalg.norm(np.maximum(np.maximum(b.low-p,p-b.high),0))>=.030 for p in hands.values())
    return bool(np.linalg.norm(o[:2]-RETURN_XY)<=.040 and TOP-.003<=low[2]<=TOP+.004
                and np.linalg.norm(b.lin_vel)<.010 and np.linalg.norm(b.ang_vel)<.10 and clear)

def score(objects):
    p,c=_fractions(objects)
    history=_MEMORY if TEST_HISTORY is None else TEST_HISTORY
    if not history.get('seated',False) or history.get('gripper_contact',False):return 0.
    if any(objects[n].low[2]<TOP-.020 for n in ('brush','cap')):return 0.
    a=p
    if p>=.95 and returned(objects):return 1.
    if a>=.75:return .75
    if a>=.5:return .5
    if a>0:return .25
    return 0.

def success(objects):return score(objects)==1.


def _begin_episode():
    if _MEMORY.get('episode')==_RUN['episode']:return
    _MEMORY.clear();_READS.clear()
    _MEMORY.update(episode=_RUN['episode'],step=None,post=np.zeros(72),clamp=np.zeros(69),previous={},contact={},seated=False,seat_since=None,gripper_contact=False,contact_events=0)
    _set_patch_colours(set(),reset=True)

def _record_gripper_contacts(actor0,actor1,positions):
    """Latch any actual gripper-link contact on the taper, even after it is clean."""
    pair=(str(actor0).split('/'),str(actor1).split('/'))
    links={'gripper','tip_left','tip_right'}
    if not any('post' in pair[i] and links.intersection(pair[1-i]) for i in (0,1)):return
    for point in positions:
        z=float(point[2])-POST_POS[2]
        if 0.<=z<=POST_HEIGHT:
            _MEMORY['gripper_contact']=True
            return


def _install_contact_reports():
    """Observe PhysX contacts at every physics step; never change collision response.
    The robot spawns with zero-threshold contact reporting on every link; Isaac Lab
    only switches contact processing off because this embodiment builds no sensor."""
    global _CONTACT_SUB
    import carb
    import isaaclab.sim as sim_utils
    from pxr import PhysicsSchemaTools,UsdPhysics
    from omni.physx import get_physx_simulation_interface
    carb.settings.get_settings().set_bool('/physics/disableContactProcessing',False)
    stage=sim_utils.get_current_stage();count=0
    for prim in stage.Traverse():
        if (prim.GetName() in ('gripper','tip_left','tip_right') and prim.HasAPI(UsdPhysics.RigidBodyAPI)
                and 'PhysxContactReportAPI' in prim.GetAppliedSchemas()):count+=1
    print('BAT_CONTACT_REPORTERS',count,flush=True)
    if count<6:print('BAT_CONTACT_WARNING missing gripper contact reporters',flush=True)
    def contacts(headers,data):
        for h in headers:
            _MEMORY['contact_events']=_MEMORY.get('contact_events',0)+h.num_contact_data
            # Collider paths retain the post and link ancestry even for static bodies.
            a=PhysicsSchemaTools.intToSdfPath(h.collider0)
            b=PhysicsSchemaTools.intToSdfPath(h.collider1)
            start=h.contact_data_offset
            _record_gripper_contacts(a,b,(data[i].position for i in range(start,start+h.num_contact_data)))
    _CONTACT_SUB=get_physx_simulation_interface().subscribe_contact_report_events(contacts)


def _record_seating(bo,br):
    # Mouth at least 8 mm below the 150 mm post top, within 3 mm coaxial
    # offset and 10 degrees, continuously for 1 s. This is visible on video.
    seated=(np.linalg.norm(bo[:2]-POST_POS[:2])<=.003
            and POST_POS[2]-.003<=bo[2]<=POST_POS[2]+.010
            and float(br[2,2])>=math.cos(math.radians(10)))
    if seated:
        if _MEMORY['seat_since'] is None:_MEMORY['seat_since']=_RUN['t']
        if _RUN['t']-_MEMORY['seat_since']>=1.:_MEMORY['seated']=True
    else:_MEMORY['seat_since']=None


def _point_data(kind,objects):
    if kind=='post':
        o,R=POST_POS,np.eye(3);indices=range(72)
    else:
        o,R=cable_frame(objects['cable']);indices=[i for i in range(72) if i%24!=0]
    pts=[]
    for idx in indices:
        z=(idx//24+.5)*.006;r=.00895-z/18;a=(idx%24+.5)*2*math.pi/24
        pts.append(o+R@np.array((r*math.cos(a),r*math.sin(a),z)))
    return np.array(pts),list(indices),R

def _update_cleaning(objects):
    if _RUN['t'] is None:return
    _begin_episode()
    if _MEMORY['step']==_RUN['step']:return
    _MEMORY['step']=_RUN['step'];b=objects['brush'];bo=np.asarray(b.origin);br=b.rotation()
    _record_seating(bo,br)
    changed=set()
    for kind,names,k in (('post',BRISTLE_NAMES,BRISTLE_K),):
        points,indices,surface_R=_point_data(kind,objects);q=(points-bo)@br
        forces=np.array([abs(float(b.joints[n]))*k for n in names])
        az=np.arctan2(q[:,1],q[:,0]);sector=(np.floor((az+math.pi/4)/(math.pi/2)).astype(int)%4)
        r=np.linalg.norm(q[:,:2],axis=1)
        if kind=='post':
            inner=np.interp(q[:,2],[0,.001,.020],[.0125,.0086,.0076])
            depth=(q[:,2]>=.001)&(q[:,2]<=.020)
            contact=depth&(np.abs(r-inner)<=.0012)&(forces[sector]>=.25)&(forces.sum()>=2.)
            alignment=float(br[:,2]@surface_R[:,2])>math.cos(math.radians(10))
        else:
            outer=.0125-(q[:,2]-.055)*(.005/.065)
            depth=(q[:,2]>=.055)&(q[:,2]<=.120)
            contact=depth&(r<=outer+.0007)&(r>=outer-.0045)&(forces[sector]>=.25)&(forces.sum()>=2.)
            alignment=float(br[:,2]@surface_R[:,2])>math.cos(math.radians(12))
        contact &= alignment
        prev=_MEMORY['previous'].get(kind);oldcontact=_MEMORY['contact'].get(kind)
        if prev is not None:
            if kind=='post':
                # A push-on alone cannot clean: require tangential sweep.
                a0=np.arctan2(prev[:,1],prev[:,0]);da=(az-a0+math.pi)%(2*math.pi)-math.pi
                ds=np.abs(da)*r
            else:ds=np.linalg.norm(q-prev,axis=1)
            ds=np.where(ds<=.010,ds,0.)  # reject discontinuities / reset jumps
            active=contact&oldcontact
            before=_MEMORY[kind]>=.020
            _MEMORY[kind][active]+=ds[active]
            after=_MEMORY[kind]>=.020
            changed.update((kind,indices[i]) for i in np.flatnonzero(after&~before))
        _MEMORY['previous'][kind]=q.copy();_MEMORY['contact'][kind]=contact.copy()
    if changed:_set_patch_colours(changed)
    if DIAG and _RUN['step']%20==0:
        co=np.zeros(3);p,c=(np.mean(_MEMORY[k]>=.020) for k in ('post','clamp'))
        print('BAT_DIAG',round(_RUN['t'],2),'seated',_MEMORY['seated'],'gripper_contact',_MEMORY['gripper_contact'],'contact_events',_MEMORY['contact_events'],'clean',round(p,3),round(c,3),'brush',np.round((bo-[0,0,TOP])*1000,1),'clamp',np.round((co-[0,0,TOP])*1000,1),'cupN',round(sum(abs(b.joints[n])*BRISTLE_K for n in BRISTLE_NAMES),2),'coneN',round(sum(abs(b.joints[n])*CONE_K for n in CONE_NAMES),2),flush=True)

def _set_patch_colours(changed,reset=False):
    # Visual material only. Collider shape, mass, contacts and poses never change.
    if _RUN['env'] is None:return
    import isaaclab.sim as sim_utils
    from pxr import Gf
    from yam_arena.station import linear_rgb
    stage=sim_utils.get_current_stage()
    if not _PAINT:
        for prim in stage.Traverse():
            path=str(prim.GetPath())
            for kind in ('post','clamp'):
                token='mesh_'+kind+'patch_'
                if token not in path:continue
                tail=path.split(token,1)[1].split('_look',1)[0]
                if not tail.isdigit():continue
                attr=prim.GetAttribute('inputs:diffuseColor')
                if attr:_PAINT.setdefault((kind,int(tail)),[]).append(attr)
    for key,attrs in _PAINT.items():
        if reset or key in changed:
            colour=COATING if reset else METAL
            for attr in attrs:attr.Set(Gf.Vec3f(*linear_rgb(colour)))

# Full-size products and the constrained service-bench layout.
BATTERY_CENTRE=(.307,-.147)
TRAY_THICK=.003
BATTERY=Part('battery',(BATTERY_CENTRE[0],BATTERY_CENTRE[1],TOP+TRAY_THICK),(
    box((0,0,.055),(.168,.175,.110)),box((0,0,.1145),(.168,.175,.009),color=(.56,.10,.08)),
    box((0,.080,.123),(.168,.015,.008),color=(.56,.10,.08)),
    box((-.076,0,.123),(.016,.145,.008),color=(.56,.10,.08)),
    box((.076,0,.123),(.016,.145,.008),color=(.56,.10,.08)),
    cyl((-.067,-.073,.124),.015,.010,color=(.14,.15,.16)),
    cyl((.067,-.073,.138),.014,.030,color=(.70,.09,.07)),
),color=(.12,.13,.14),contact_offset=.0005)
TRAY=Part('tray',(BATTERY_CENTRE[0],BATTERY_CENTRE[1],TOP),(
    box((0,0,.0015),(.2794,.1778,.003)),box((0,.0899,.012),(.2854,.003,.024)),
    box((0,-.0899,.012),(.2854,.003,.024)),box((.1412,0,.012),(.003,.1838,.024)),box((-.1412,0,.012),(.003,.1838,.024)),
    box((0,0,.132),(.025,.175,.003),color=(.09,.10,.11)),
),color=(.09,.10,.11),contact_offset=.0005)
BRUSH=brush(BRUSH_POS)
CAP=cap(CAP_PLACE)
POST=painted_post(POST_POS)
OBJECTS=[TRAY,BATTERY,POST,BRUSH,CAP]

# Scripted demonstration: all targets drive ordinary gripper contact.
G=grip_rotation(0)
SIDE=np.array([[0.,0,1],[1,0,0],[0,1,0]])
a=math.pi/4
RY=np.array([[math.cos(a),0,math.sin(a)],[0,1,0],[-math.sin(a),0,math.cos(a)]])
SIDE45=RY@SIDE@np.diag([-1.,-1.,1.])
class Grip(float):
    def __new__(cls,yaw,rot):
        obj=super().__new__(cls,yaw);obj.rot=np.asarray(rot);return obj

def _install_grip():
    import yam_arena.demo as dm
    old=getattr(dm.grip_rotation,'__wrapped__',dm.grip_rotation)
    def rotation(yaw=0.):return yaw.rot.copy() if isinstance(yaw,Grip) else old(float(yaw))
    rotation.__wrapped__=old;dm.grip_rotation=rotation
_install_grip()


def _read(name):
    if _RUN['env'] is None:return None
    from yam_arena.scene import object_states
    return object_states(_RUN['env'],[name])[name]

class W(Waypoint):
    def __init__(self,secs,p,f,label,rot=None,read=None):
        self.duration,self._p,self.finger,self.label=secs,np.asarray(p,float),f,label
        self.yaw=0. if rot is None else (rot if isinstance(rot,Grip) else Grip(0.,rot))
        self.read=read
    @property
    def pos(self):
        if self.read and _RUN['env'] is not None:
            key,name,nominal=self.read
            if key not in _READS:
                s=_read(name)
                if s is not None:_READS[key]=np.asarray(s.origin)-np.asarray(nominal)
            return tuple(self._p+_READS.get(key,0.))
        return tuple(self._p)

class PostWaypoint(W):
    """Expert cup-height feedback with <=11 mm compliance lead, no pose writes.

    The offline pose is the geometric contact pose. Runtime compensates arm
    give from the visually observed cup height, capped to a modest lead.
    """
    @property
    def pos(self):
        if _RUN['env'] is None:return tuple(self._p)
        b=_read('brush')
        if b is None:return tuple(self._p)
        key='post_height_servo'
        state=_READS.setdefault(key,{'step':None,'z':float(self._p[2])})
        if state['step']!=_RUN['step']:
            error=POST_POS[2]+.001-float(b.origin[2])
            state['z']=float(np.clip(state['z']+.15*error,self._p[2]-.011,self._p[2]+.002))
            state['step']=_RUN['step']
        return (float(self._p[0]),float(self._p[1]),state['z'])

class WithdrawWaypoint(W):
    """Twist while lifting slowly, reducing axial bristle drag on the pinch."""
    def __init__(self,secs,p,f,label,rot,fraction):
        super().__init__(secs,p,f,label,rot)
        self.fraction=fraction
    @property
    def pos(self):
        if _RUN['env'] is None:return tuple(self._p)
        z=_READS.setdefault('withdraw_start_z',_READS.get('post_height_servo',{}).get('z',POST_POS[2]+GRIP+.001))
        target=POST_POS[2]+GRIP+.032
        return (float(self._p[0]),float(self._p[1]),float(z+(target-z)*self.fraction))

def demo():
    left=[];right=[];t=0.
    parkL=np.array((.15,.32,TOP+.13));parkR=np.array((.23,-.365,TOP+.18))
    pl,pr=parkL.copy(),parkR.copy();fl=fr=FINGER_OPEN;rl=rr=None
    def go(s,label,lp=None,lf=None,lr='keep',rp=None,rf=None,rrr='keep',readL=None,readR=None):
        nonlocal t,pl,pr,fl,fr,rl,rr
        if lp is not None:pl=np.asarray(lp,float)
        if rp is not None:pr=np.asarray(rp,float)
        if lf is not None:fl=lf
        if rf is not None:fr=rf
        if not isinstance(lr,str):rl=lr
        if not isinstance(rrr,str):rr=rrr
        left.append(W(s,pl,fl,label,rl,readL));right.append(W(s,pr,fr,label,rr,readR));t+=s
    pick=BRUSH_POS+[0,0,GRIP];readbody=('bodypick','brush',BRUSH_POS)
    go(2.,'approach uncapped brush',rp=pick+[0,0,.09],readR=readbody)
    go(1.2,'down around plastic body',rp=pick,readR=readbody)
    go(.8,'grasp body 32 mm',rf=PINCH,readR=readbody)
    go(1.6,'lift cup clear',rp=pick+[0,0,.150],readR=readbody)
    seat=POST_POS+[0,0,GRIP+.001];hover=POST_POS+[0,0,GRIP+.032]
    go(2.,'cup square above battery post',rp=hover)
    go(2.,'press cup over taper',rp=seat)
    right[-1]=PostWaypoint(2.,seat,PINCH,'press cup over taper')
    for i in range(12):
        angle=0. if DEMO_PRESS_ONLY else math.radians(40 if i%2==0 else -40)
        go(1.2,'loaded post twist',rp=seat,rrr=Grip(angle,grip_rotation(angle)))
        right[-1]=PostWaypoint(1.2,seat,PINCH,'loaded post twist',Grip(angle,grip_rotation(angle)))
    for i in range(10):
        angle=0. if DEMO_PRESS_ONLY else math.radians(40 if i%2==0 else -40)
        target=seat+(hover-seat)*(i+1)/10
        turn=Grip(angle,grip_rotation(angle))
        go(1.,'twist while withdrawing cup',rp=target,rrr=turn)
        right[-1]=WithdrawWaypoint(1.,target,PINCH,'twist while withdrawing cup',turn,(i+1)/10)
    go(.8,'cup clear and upright',rp=hover,rrr=None)
    go(2.,'carry brush clear of battery before descent',rp=(.19,-.22,TOP+.200))
    if DEMO_PRESS_ONLY:
        returnp=np.array((RETURN_XY[0],RETURN_XY[1],TOP+.056))
        go(2.,'carry unworked brush back',rp=returnp+[0,0,.100])
        go(1.5,'stand brush after pressing only',rp=returnp)
        go(.8,'release brush after wrong way',rf=FINGER_OPEN)
        go(1.5,'retreat after wrong way',rp=parkR)
        go(2.,'wrong-way final view')
        return {'left':left,'right':right}
    returnp=np.array((RETURN_XY[0],RETURN_XY[1],TOP+.056))
    go(2.,'carry brush back',rp=returnp+[0,0,.100])
    go(1.5,'stand brush on bench',rp=returnp)
    go(.8,'release brush',rf=FINGER_OPEN)
    go(1.5,'retreat clear of brush',rp=parkR)
    go(2.,'final still view')
    return {'left':left,'right':right}
