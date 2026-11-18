"""Whole-trial tool-use and contact shortcut regressions."""
import numpy as np
from test_clean_battery_post_and_cable_clamp_with_terminal_brush import T, GOAL, _rot


def test_fingertip_rub_shortcut_is_zero(monkeypatch):
    monkeypatch.setattr(T,'TEST_COVERAGE',[1.,0.])
    # Even a magically clean post with the untouched returned brush gets no credit.
    monkeypatch.setattr(T,'TEST_HISTORY',{'seated':False,'gripper_contact':False})
    assert T.score(GOAL)==0.
    # Seating the tool first cannot excuse later rubbing with any gripper link.
    monkeypatch.setattr(T,'TEST_HISTORY',None)
    monkeypatch.setattr(T,'_MEMORY',{'seated':True,'gripper_contact':False})
    for link in ('gripper','tip_left','tip_right'):
        T._MEMORY['gripper_contact']=False
        T._record_gripper_contacts('/World/post',f'/World/right_robot/{link}',
                                   [T.POST_POS+np.array([.0085,0,.009])])
        assert T._MEMORY['gripper_contact']
        assert T.score(GOAL)==0.
        assert not T.success(GOAL)


def test_contact_filter_and_latched_failure(monkeypatch):
    monkeypatch.setattr(T,'_MEMORY',{'seated':True,'gripper_contact':False})
    T._record_gripper_contacts('/World/post','/World/brush',[T.POST_POS+[0,0,.009]])
    T._record_gripper_contacts('/World/post','/World/left_robot/tip_left',[T.POST_POS+[0,0,-.002]])
    assert not T._MEMORY['gripper_contact']
    T._record_gripper_contacts('/World/left_robot/tip_right','/World/post',[T.POST_POS+[0,0,.009]])
    T._record_gripper_contacts('/World/post','/World/brush',[])
    assert T._MEMORY['gripper_contact']


def test_seating_needs_depth_alignment_and_one_continuous_second(monkeypatch):
    monkeypatch.setattr(T,'_MEMORY',{'seated':False,'seat_since':None})
    monkeypatch.setattr(T,'_RUN',{'t':0.})
    T._record_seating(T.POST_POS+[0,0,.001],np.eye(3))
    T._RUN['t']=.9
    T._record_seating(T.POST_POS+[0,0,.001],np.eye(3))
    assert not T._MEMORY['seated']
    T._record_seating(T.POST_POS+[.004,0,.001],np.eye(3))
    assert T._MEMORY['seat_since'] is None
    T._RUN['t']=1.
    T._record_seating(T.POST_POS+[0,0,.011],np.eye(3))
    assert T._MEMORY['seat_since'] is None
    T._record_seating(T.POST_POS+[0,0,.001],_rot(0,11,'x'))
    assert T._MEMORY['seat_since'] is None
    T._record_seating(T.POST_POS+[0,0,.001],np.eye(3))
    T._RUN['t']=2.
    T._record_seating(T.POST_POS+[0,0,.001],np.eye(3))
    assert T._MEMORY['seated']
    T._record_seating(T.BRUSH_POS,np.eye(3))
    assert T._MEMORY['seated']
