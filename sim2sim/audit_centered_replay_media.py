"""Decode the two saved-pose replay deliverables; no physics or model edits."""
import json
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
import shared as s

rows=[]
sheet=Image.new('RGB',(1280,840),'#eeeeee')
for column,device in enumerate(['cpu','gpu']):
    path=s.ROOT/f'demos/common_contact_centered_{device}_w2p5_B_closeup_replay.mp4'
    with imageio.get_reader(path) as reader:
        metadata=reader.get_meta_data()
        assert metadata['size']==(1280,800) and metadata['fps']==25
        frames=0
        for i,frame in enumerate(reader):
            assert frame.shape==(800,1280,3) and frame.dtype==np.uint8
            if i in [0,75,150]:
                thumbnail=Image.fromarray(frame).resize((640,400))
                # Two rows show actual startup and six-second terminal poses.
                if i in [0,150]:sheet.paste(thumbnail,(640*column,40+400*(i//150)))
            frames+=1
        assert frames==176,frames
    provenance=json.loads(path.with_suffix('.json').read_text())
    assert not provenance['physics_recomputed'] and not provenance['native_isaac_renderer']
    rows.append(dict(device=device,path=str(path),sha256=s.sha(path),frame_count=frames,
                     duration_s=frames/25,codec=metadata['codec'],size=metadata['size'],fps=25,
                     source_runs=provenance['source_runs'],fully_decoded=True))
ImageDraw.Draw(sheet).text((20,12),'CPU comparison: startup / final pose',fill='black')
ImageDraw.Draw(sheet).text((660,12),'GPU comparison: startup / final pose',fill='black')
sheet.save(s.ROOT/'audit/centered_replay_media_contact_sheet.png')
s.write(s.ROOT/'audit/centered_replay_media.json',dict(passed=True,rows=rows,source_sha=s.sha(__file__)))
print(json.dumps(dict(passed=True,rows=rows)))
