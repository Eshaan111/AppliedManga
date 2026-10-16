"""CPU-only tests: no model imports or GPU inference."""
import argparse
import ast
import contextlib
import copy
import importlib.util
import io
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / 'src/artifact_contour_sequence.py'
BASELINE = ROOT / 'LegacyMagiPipelines/artifact_contour_sequence_per_sequence.py'


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def functions(path):
    return {node.name: node for node in ast.parse(path.read_text(encoding='utf-8-sig')).body if isinstance(node, ast.FunctionDef)}


class RemoveLoadingWrappers(ast.NodeTransformer):
    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == 'load_model_once':
            return node.args[1].body
        return node


class NormalizeCacheLogging(ast.NodeTransformer):
    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == 'cache_message':
            status = node.value.args[0].value
            return ast.parse("print(f'RUN   {name}')" if status == 'MISS' else "print(f'CACHE {name}')").body[0]
        return self.generic_visit(node)


class WorkflowTests(unittest.TestCase):
    def test_prediction_functions_and_prompts_preserved(self):
        before, after = functions(BASELINE), functions(CURRENT)
        for name, node in before.items():
            if name == 'cached_stage':
                continue
            normalized = RemoveLoadingWrappers().visit(copy.deepcopy(after[name]))
            self.assertEqual(ast.dump(node), ast.dump(normalized), name)
        # Cache compatibility now also handles the known folder rename.
        cache_source=ast.unparse(after['cached_stage'])
        self.assertIn("digest(candidate['data']) == candidate['data_hash']",cache_source)
        self.assertIn('ALLOWED_STAGES',cache_source)
        self.assertIn('normalize_cache_paths',cache_source)
        def assignments(path):
            output = {}
            for node in ast.parse(path.read_text(encoding='utf-8-sig')).body:
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id in {'SYSTEM_PROMPT','EXAMPLES','GEMMA_EXAMPLES','ENTITY_THRESHOLD','THRESHOLD_CHECKPOINT_EPOCH','DETECTION'}:
                            output[target.id] = ast.dump(node.value)
            return output
        self.assertEqual(assignments(BASELINE), assignments(CURRENT))

    def test_folder_rename_cache_reuse_keeps_integrity_checks(self):
        import hashlib
        from datetime import datetime,timezone
        nodes=functions(CURRENT)
        names=['digest','write_json','normalize_cache_paths','cached_stage','stage_ref']
        ns=dict(Path=Path,json=json,hashlib=hashlib,datetime=datetime,timezone=timezone,
                json_ready=lambda value:value,MAGI_ROOT=ROOT.parent,
                cache_message=lambda *args:None)
        exec(compile(ast.Module(body=[nodes[name] for name in names],type_ignores=[]),'cache_test','exec'),ns)
        inputs=dict(sequence_id='example',pages=[dict(sha256='same-image')])
        old_settings=dict(checkpoint=dict(path=str(ROOT.parent.parent/'Magi'/'weights.pt'),sha256='same-weights'))
        settings=ns['normalize_cache_paths'](old_settings)
        payload=dict(signature=ns['digest'](dict(inputs=inputs,algorithm_version=1,settings=old_settings)),
                     data=[dict(text='HELLO')],created_at='saved')
        payload['data_hash']=ns['digest'](payload['data'])
        manifest=dict(stages=dict(embeddings=dict(settings=old_settings,algorithm_version=1)))
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            (folder/'embeddings.json').write_text(json.dumps(payload),encoding='utf-8')
            ns.update(INPUTS=inputs,STAGE_VERSIONS={'embeddings':1},CACHE_DIR=folder,
                      MANIFEST_PATH=folder/'manifest.json',manifest=manifest,
                      FORCE={'embeddings':False},ALLOWED_STAGES=set())
            def forbidden():raise AssertionError('Unexpected inference')
            self.assertEqual(ns['cached_stage']('embeddings',settings,forbidden),payload['data'])
            self.assertEqual(ns['stage_ref']('embeddings')['signature'],payload['signature'])
            self.assertEqual(ns['cached_stage']('embeddings',settings,forbidden),payload['data'])
            changed=copy.deepcopy(settings);changed['checkpoint']['sha256']='different-weights'
            with self.assertRaises(RuntimeError):ns['cached_stage']('embeddings',changed,forbidden)
            damaged=copy.deepcopy(payload);damaged['data_hash']='wrong'
            (folder/'embeddings.json').write_text(json.dumps(damaged),encoding='utf-8')
            with self.assertRaises(RuntimeError):ns['cached_stage']('embeddings',settings,forbidden)

    def test_raw_transcript_and_routing_on_saved_data(self):
        outputs = next((ROOT/'cache_artifact_contour_full_inference').glob('*/*/outputs'), None)
        if outputs is None:
            self.skipTest('No local saved example')
        groups = json.loads((outputs/'group_predictions.json').read_text())
        entities = json.loads((outputs/'entities.json').read_text())['entities']
        original = json.loads((outputs/'transcript_raw.json').read_text())
        pages = [Path(str(i)) for i in range(3)]
        results = []
        for source in [BASELINE, CURRENT]:
            nodes=functions(source); ns={}
            exec(compile(ast.Module(body=[nodes['build_transcript'],nodes['resolve_group_speaker']],type_ignores=[]),'routing_test','exec'),ns)
            transcript,_ = ns['build_transcript'](original['sequence_id'],groups,pages,pages,entities)
            self.assertEqual(transcript,original)
            def member(bid,ids):return dict(box_id=bid,entity_ids=ids,character_links=[],association_status='linked' if ids else 'unassociated')
            a,b,c = member('T1',['E1']),member('T2',[]),member('T3',['E2'])
            results.append([ns['resolve_group_speaker'](role,members) for role,members in [('NARRATED',[a]),('SPOKEN',[a]),('SPOKEN',[b]),('SPOKEN',[a,c]),('SPOKEN',[a,b])]])
        self.assertEqual(results[0],results[1])

    def fixture_worker(self, folder):
        current=functions(CURRENT)
        setup='''import json,hashlib
from datetime import datetime,timezone
from pathlib import Path
SHARED_MODELS=globals().get('SHARED_MODELS')
ALLOWED_STAGES=globals().get('ALLOWED_STAGES')
'''+ast.unparse(current['load_model_once'])+'''
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
def json_ready(value):return value
def write_json(path,value):path.write_text(json.dumps(value),encoding='utf-8')
def event(value):
    with Path(BATCH_SETTINGS['event_file']).open('a',encoding='utf-8') as stream:stream.write(value+'\\n')
def create(name):
    event('load:'+name)
    return {'patched':False}
'''
        settings='''SID=BATCH_SETTINGS['sequence_id']
CACHE_DIR=Path(BATCH_SETTINGS['cache_root'])/SID
CACHE_DIR.mkdir(parents=True,exist_ok=True)
OUTPUT_DIR=Path(BATCH_SETTINGS['sequence_output_dir'])
INPUTS={'sid':SID}
STAGES=['magi_raw','embeddings','paddle_ocr','ocr_selection','artifact_labels','entities','contour_bubbles','contour_roles','contour_speakers','gemma_transcript']
STAGE_VERSIONS={name:1 for name in STAGES}
FORCE={name:BATCH_SETTINGS.get('force',False) and name in ALLOWED_STAGES for name in STAGES}
manifest={'stages':{}}
MANIFEST_PATH=CACHE_DIR/'manifest.json'
'''
        helpers=ast.unparse(current['cache_message'])+'\n'+"checkpoint=load_model_once('character_checkpoint',lambda:create('character_checkpoint'))\n"+ast.unparse(current['cached_stage'])+'\n'
        predictors='''magi_model=None
def get_magi():return load_model_once('magi',lambda:create('magi'))
def predict_magi():
    assert not get_magi()['patched'],'Original Magi ran after patching'
    event('raw:'+SID)
    return {'text':SID}
def validate_magi(value):assert value['text']==SID
def predict_embeddings():
    get_magi()['patched']=True
    event('embeddings:'+SID)
    return ['E1',SID]
'''
        cell12='''try:
    magi_raw_results=cached_stage('magi_raw',{},predict_magi)
    validate_magi(magi_raw_results)
    embeddings=cached_stage('embeddings',{},predict_embeddings)
finally:
    magi_model=None
'''
        paddle="paddle_result=cached_stage('paddle_ocr',{},lambda:(load_model_once('paddle',lambda:create('paddle')) and magi_raw_results))\n"
        artifacts="ocr_selection=cached_stage('ocr_selection',{},lambda:paddle_result)\nartifact_result=cached_stage('artifact_labels',{},lambda:(load_model_once('gemma_tokenizer',lambda:create('gemma_tokenizer')) and load_model_once('gemma',lambda:create('gemma')) and ocr_selection))\n"
        entities="entities=cached_stage('entities',{},lambda:embeddings)\nbubbles=cached_stage('contour_bubbles',{},lambda:(load_model_once('yolo',lambda:create('yolo')) and artifact_result))\n"
        contour="roles=cached_stage('contour_roles',{},lambda:bubbles)\nrouting=cached_stage('contour_speakers',{},lambda:roles)\n"
        cleanup="cleaned=cached_stage('gemma_transcript',{},lambda:(load_model_once('gemma_tokenizer',lambda:create('gemma_tokenizer')) and load_model_once('gemma',lambda:create('gemma')) and routing))\ntranscript={'sequence_id':SID,'pages':[[{'speaker':'character1','text':cleaned['text']}],[],[]]}\nwrite_json(OUTPUT_DIR/'transcript.json',transcript)\n"
        pieces=[(2,setup),(4,settings),(6,''),(8,helpers),(10,predictors),(12,cell12),(16,paddle),(20,artifacts),(32,entities),(42,contour),(44,cleanup)]
        worker=folder/'fixture.py'
        worker.write_text('\n'.join(f'# --- Notebook inference cell {number} ---\n{code}' for number,code in pieces),encoding='utf-8')
        return worker

    def test_stage_order_reuse_and_force_refresh(self):
        scheduler=load_module('scheduler_test',ROOT/'src/artifact_contour_batch.py')
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            folder=Path(directory);worker=self.fixture_worker(folder);events_path=folder/'events.txt'
            configs=[dict(sequence_id=sid,images=[],split='test',magi_root=str(ROOT.parent),cache_root=str(folder/'cache'),sequence_output_dir=str(folder/sid),event_file=str(events_path),force=True) for sid in ['seq_a','seq_b']]
            with contextlib.redirect_stdout(io.StringIO()):scheduler.run_stages(configs,folder/'status.json',worker)
            events=events_path.read_text().splitlines()
            self.assertEqual(Counter(x[5:] for x in events if x.startswith('load:')),Counter(magi=1,character_checkpoint=1,paddle=1,gemma=2,gemma_tokenizer=2,yolo=1))
            self.assertLess(events.index('raw:seq_b'),events.index('embeddings:seq_a'))
            status=json.loads((folder/'status.json').read_text())
            self.assertEqual(status['status'],'complete')
            self.assertEqual(status['completed_phases'],[phase[0] for phase in scheduler.PHASES])
            for config in configs:
                actual=json.loads((Path(config['sequence_output_dir'])/'transcript.json').read_text())
                self.assertEqual(actual,{'sequence_id':config['sequence_id'],'pages':[[{'speaker':'character1','text':config['sequence_id']}],[],[]]})
            # Compare full sequential execution with staged execution on identical fixture inputs.
            cells,_=scheduler.load_cells(worker)
            for config in configs:
                sequential=dict(config,cache_root=str(folder/'sequential_cache'),sequence_output_dir=str(folder/'sequential'/config['sequence_id']))
                Path(sequential['sequence_output_dir']).mkdir(parents=True)
                ns=dict(BATCH_SETTINGS=sequential,SHARED_MODELS={},ALLOWED_STAGES=set().union(*(phase[2] for phase in scheduler.PHASES)))
                with contextlib.redirect_stdout(io.StringIO()):
                    for code in cells.values():exec(code,ns)
                sequential_result=json.loads((Path(sequential['sequence_output_dir'])/'transcript.json').read_text())
                staged_result=json.loads((Path(config['sequence_output_dir'])/'transcript.json').read_text())
                self.assertEqual(sequential_result,staged_result)
                ns.clear()
            events_path.write_text('')
            for config in configs:config['force']=False
            with contextlib.redirect_stdout(io.StringIO()):scheduler.run_stages(configs,folder/'status.json',worker)
            self.assertEqual(events_path.read_text().splitlines(),['load:character_checkpoint'])

    def test_missing_prior_stage_guard(self):
        scheduler=load_module('guard_test',ROOT/'src/artifact_contour_batch.py')
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            folder=Path(directory);worker=self.fixture_worker(folder);cells,_=scheduler.load_cells(worker)
            config=dict(sequence_id='seq_a',cache_root=str(folder/'cache'),sequence_output_dir=str(folder/'out'),event_file=str(folder/'events'))
            ns=dict(BATCH_SETTINGS=config,SHARED_MODELS={},ALLOWED_STAGES={'paddle_ocr'})
            for number in [2,4,6,8,10]:exec(cells[number],ns)
            with self.assertRaisesRegex(RuntimeError,'Prior stage magi_raw'):
                ns['cached_stage']('magi_raw',{},lambda:self.fail('Must not rerun prior inference'))

    def test_failed_batch_preserves_previous_predictions(self):
        runner=load_module('export_test',ROOT/'run_sequence_inference.py')
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            folder=Path(directory);data=folder/'TaskData';data.mkdir();images=[]
            for i in range(3):
                page=data/f'{i}.png';page.touch();images.append(page.name)
            (data/'sequences.json').write_text(json.dumps([dict(sequence_id='seq_a',split='test',images=images)]))
            output=folder/'out';output.mkdir();previous=output/'predictions.jsonl';previous.write_text('previous complete export')
            args=argparse.Namespace(task_data=data,split='test',sequence_ids=None,limit=None,output_dir=output,cache_root=folder/'cache',dry_run=False,overwrite=True,model_config=None,no_cleanup=False,force=False)
            def fail(configs,output_dir):raise RuntimeError('simulated stage failure')
            runner.run_shared_stages=fail
            with contextlib.redirect_stdout(io.StringIO()),self.assertRaises(RuntimeError):runner.run_batch(args)
            self.assertEqual(previous.read_text(),'previous complete export')
            self.assertEqual(json.loads((output/'batch_status.json').read_text())['status'],'failed')


if __name__=='__main__':unittest.main(verbosity=2)
