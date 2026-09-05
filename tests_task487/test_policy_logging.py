import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from task487_runtime.contract import PolicyRequest
from task487_runtime.policy_logging import PolicyExchangeRecorder
from task487_runtime.worker import InferenceWorker
from task487_runtime.gripper_telemetry import feedback_rows


def records(path):
    return [json.loads(line) for line in (path/"policy_io.jsonl").read_text().splitlines()]


def wait_result(worker):
    end = time.monotonic()+2
    while time.monotonic() < end:
        result = worker.poll()
        if result is not None: return result
        time.sleep(.001)
    pytest.fail("inference result timeout")


@pytest.mark.parametrize("horizon,cameras", [(20,4),(20,5),(16,3)])
def test_exact_rgb_state_response_and_ids_survive_archival(tmp_path, horizon, cameras):
    rng = np.random.default_rng(42)
    obs = {f"cam_{i}": rng.integers(0,256,(224,224,3),dtype=np.uint8) for i in range(cameras)}
    obs.update(pre_state=np.arange(20,dtype=np.float32), state=np.arange(20,dtype=np.float32)+1,
               prompt="Vegetable and Fruit Sorting", action_prefill_len=np.int32(3),
               actions=rng.random((horizon,20)), fixed_head_mask=np.zeros((224,224),np.uint8))
    original = {k:np.array(v,copy=True) for k,v in obs.items()}
    raw = np.zeros((horizon,20),np.float32)
    raw[:,3:9]=[1,0,0,0,1,0];raw[:,13:19]=[1,0,0,0,1,0]
    raw[:,9]=.9  # Keep raw output beyond physical range for diagnosis.
    def infer(actual):
        for k,v in original.items(): np.testing.assert_array_equal(actual[k],v)
        return {"actions":raw,"timing":{"inference_ms":1.2}}
    request=PolicyRequest(obs,12.08,np.zeros((2,6)),round_id=7,diagnostics={"timestamps":[12.,12.08]})
    with PolicyExchangeRecorder(tmp_path,min_free_bytes=0) as logger:
        with InferenceWorker(SimpleNamespace(infer=infer),horizon,logger) as worker:
            for expected_id in (1,2):
                worker.submit(request);result=wait_result(worker)
                assert result.error is None and result.request.request_id==expected_id
                np.testing.assert_allclose(result.targets[:,6],np.rad2deg(raw[:,9]))
                worker.record_disposition(result,"accepted" if expected_id==1 else "stale_round")
    rows=records(tmp_path)
    for row in rows:
        if row['event']=='request':
            arr=np.load(tmp_path/row['array_file'],allow_pickle=False)
            for k,v in original.items():np.testing.assert_array_equal(arr[f'observation/{k}'],v)
            assert row['round_id']==7 and row['observation_time']==12.08
        if row['event']=='response':
            arr=np.load(tmp_path/row['array_file'],allow_pickle=False)
            np.testing.assert_array_equal(arr['output/actions'],raw)
            assert arr['output/actions'].dtype==np.float32
    assert len([r for r in rows if r['event']=='response'])==2
    status=json.loads((tmp_path/'policy_logging_status.json').read_text())
    assert status['dropped']==status['errors']==0 and status['state']=='closed'


@pytest.mark.parametrize("invalid", [False, True])
def test_request_and_raw_error_response_survive_failed_inference(tmp_path,invalid):
    def infer(_):
        if not invalid: raise RuntimeError('synthetic network failure')
        return {'actions':np.full((20,20),np.nan,np.float32)}
    with PolicyExchangeRecorder(tmp_path,min_free_bytes=0) as logger:
        with InferenceWorker(SimpleNamespace(infer=infer),recorder=logger) as worker:
            worker.submit(PolicyRequest({'state':np.zeros(20),'prompt':'test'},1,np.zeros((2,6))))
            result=wait_result(worker);assert result.error is not None
            worker.record_disposition(result,'inference_error')
    rows=records(tmp_path)
    response=next(r for r in rows if r['event']=='response')
    assert response['error'] and next(r for r in rows if r['event']=='request')['array_file']
    if invalid:
        assert np.isnan(np.load(tmp_path/response['array_file'])['output/actions']).all()


def test_blocked_writer_never_blocks_producer_and_counts_overflow(tmp_path):
    entered,release=threading.Event(),threading.Event()
    class Blocked(PolicyExchangeRecorder):
        def _write_item(self,*args):
            if args[0]=='request':entered.set();assert release.wait(2)
            return super()._write_item(*args)
    logger=Blocked(tmp_path,queue_size=1,min_free_bytes=0)
    try:
        logger.record('request',1,arrays={'state':np.zeros(20)})
        assert entered.wait(1)
        logger.record('disposition',1,status='accepted')
        logger.record('disposition',2,status='stale_round')
        assert logger._stats['dropped']==1
    finally:release.set();logger.close()
    assert json.loads((tmp_path/'policy_logging_status.json').read_text())['last_dropped_request_id']==2


def test_image_budget_keeps_numeric_evidence_and_marks_missing_rgb(tmp_path):
    with PolicyExchangeRecorder(tmp_path,image_budget_bytes=0,min_free_bytes=0) as logger:
        logger.record('request',1,arrays={'observation':{'cam_left_top':np.zeros((224,224,3),np.uint8),
                                                        'state':np.ones(20),'prompt':'test'}})
    row=next(r for r in records(tmp_path) if r['event']=='request')
    assert row['omitted_image_keys']==['observation/cam_left_top']
    data=np.load(tmp_path/row['array_file'])
    assert 'observation/state' in data and 'observation/cam_left_top' not in data


def test_disk_failure_does_not_drop_policy_result(tmp_path):
    class Failing(PolicyExchangeRecorder):
        def _write_item(self,*args):
            if args[0]=='response':raise OSError('synthetic full disk')
            return super()._write_item(*args)
    raw=np.zeros((20,20));raw[:,3:9]=raw[:,13:19]=[1,0,0,0,1,0]
    with Failing(tmp_path,min_free_bytes=0) as logger:
        with InferenceWorker(SimpleNamespace(infer=lambda _: {'actions':raw}),recorder=logger) as worker:
            worker.submit(PolicyRequest({'prompt':'test'},1,np.zeros((2,6))))
            assert wait_result(worker).error is None
    status=json.loads((tmp_path/'policy_logging_status.json').read_text())
    assert status['errors']==1 and 'full disk' in status['last_error']


def test_reused_log_directory_does_not_overwrite_previous_request(tmp_path):
    paths=[]
    for value in (1,2):
        with PolicyExchangeRecorder(tmp_path,min_free_bytes=0) as logger:
            logger.record('request',1,arrays={'state':np.array([value])})
            paths.append(logger.path/'request_000001.npz')
    assert paths[0]!=paths[1]
    assert [np.load(path)['state'].item() for path in paths]==[1,2]


def test_async_snapshot_owns_pixels_before_caller_mutation(tmp_path):
    array=np.arange(100).reshape(10,10)
    with PolicyExchangeRecorder(tmp_path,min_free_bytes=0) as logger:
        logger.record('request',1,arrays={'pixels':array})
        array[:]=0
    row=next(r for r in records(tmp_path) if r['event']=='request')
    assert np.load(tmp_path/row['array_file'])['pixels'].sum()==sum(range(100))


def test_gripper_feedback_keeps_both_masters_and_motor_torque_units():
    msg=SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(sec=100,nanosec=500000000)),
        name=['Joint69','Joint610','Joint79','Joint710'],position=[-.1,.1,-.2,.2],
        velocity=[.3,-.3,.4,-.4],effort=[10.,0.,-20.,0.])
    rows=feedback_rows(msg,{'Joint79':(-.3,100.)},100.6,5.,1)
    assert [r[4] for r in rows]==['left','right']
    assert rows[0][10]==.5 and rows[1][10]==-1.
    assert rows[1][7]==pytest.approx(np.rad2deg(.2))
    assert rows[1][-1]==pytest.approx(.6)
    assert rows[0][11] is None
    msg.effort=[]
    assert all(row[9] is None and row[10] is None for row in feedback_rows(msg,{},101,6,2))
