import glob
import os
import muda
import jams


parent_dir = 'urban-sound-classification/train/Train'
file_ext="*.wav"
out_dir = 'urban-sound-classification/Augmented'

bg_files = ["background_noise/150993__saphe__street-scene-1.wav", "background_noise/173955__saphe__street-scene-3.wav"]

bg_noise = muda.deformers.BackgroundNoise(files=bg_files)
drc = muda.deformers.DynamicRangeCompression(preset=['music standard', 'film standard', 'radio', 'speech'])

for jfn in glob.glob(os.path.join(parent_dir, file_ext)):
        fn = jfn.split('/')[len(jfn.split('/'))-1].split('.')[0]
        print(jfn)
        j_orig = muda.load_jam_audio(jams.JAMS(),
                     jfn)
        for i, jam_out in enumerate(bg_noise.transform(j_orig)):
                muda.save(out_dir + '/bg/' + fn + '_{:02d}.wav'.format(i),
                          out_dir + '/bg/' + fn + '_{:02d}.jams'.format(i),jam_out)
