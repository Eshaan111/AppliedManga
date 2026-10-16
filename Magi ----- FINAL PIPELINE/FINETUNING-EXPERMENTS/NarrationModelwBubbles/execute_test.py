"""Execute the one-page notebook in the current Godfather Python process.

Captures previews into an executed notebook without requiring a Jupyter server.
"""
from pathlib import Path
import base64
import contextlib
import io
import json
import traceback

ROOT = Path(__file__).resolve().parent


def main():
    notebook = json.loads((ROOT / 'NarrationModelwBubbles_Test.ipynb').read_text(encoding='utf-8'))
    target = ROOT / 'NarrationModelwBubbles_Test_executed.ipynb'
    namespace = {'__name__': '__main__'}
    count = 0
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        count += 1
        print('Executing test cell', count, flush=True)
        outputs, stream = [], io.StringIO()

        def flush_stream():
            value = stream.getvalue()
            if value:
                outputs.append(dict(output_type='stream', name='stdout', text=value.splitlines(keepends=True)))
                stream.seek(0)
                stream.truncate(0)

        def capture_display(value):
            from PIL import Image
            flush_stream()
            if isinstance(value, Image.Image):
                buffer = io.BytesIO()
                value.save(buffer, format='PNG')
                data = {'image/png': base64.b64encode(buffer.getvalue()).decode(), 'text/plain': str(value)}
            elif hasattr(value, '_repr_html_'):
                data = {'text/html': value._repr_html_(), 'text/plain': str(value)}
            else:
                data = {'text/plain': str(value)}
            outputs.append(dict(output_type='display_data', data=data, metadata={}))

        try:
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                exec(compile(''.join(cell['source']), f'one_page_test:cell{count}', 'exec'), namespace)
                # The imports cell defines display; subsequent cells use this capture hook.
                namespace['display'] = capture_display
        except Exception as error:
            flush_stream()
            outputs.append(dict(output_type='error', ename=type(error).__name__,
                                evalue=str(error), traceback=traceback.format_exc().splitlines()))
            cell.update(outputs=outputs, execution_count=count)
            target.write_text(json.dumps(notebook, indent=1, ensure_ascii=False)+'\n', encoding='utf-8')
            raise
        flush_stream()
        cell.update(outputs=outputs, execution_count=count)
        target.write_text(json.dumps(notebook, indent=1, ensure_ascii=False)+'\n', encoding='utf-8')
    print('PASS: all one-page notebook cells executed:', target, flush=True)


if __name__ == '__main__':
    main()
