# 90%  5%  5%  data would be fine
import os
import shutil
import random


def separate(midi_dir, separations=(0.9, 0.05, 0.05)):
    assert sum(separations) == 1 and len(separations) == 3
    file_names = [f for f in os.listdir(midi_dir) if f.endswith('.mid')]
    training_sep, dev_sep = int(len(file_names) * separations[0]), int(len(file_names) * separations[1])
    random.shuffle(file_names)
    for name in ('train', 'dev', 'test'):
        dir_path = os.path.join(midi_dir, name)
        if os.path.isdir(dir_path):
            shutil.rmtree(dir_path)
        os.mkdir(dir_path)
    for file_name in file_names[0: training_sep]:
        shutil.copyfile(os.path.join(midi_dir, file_name), os.path.join(midi_dir, 'train', file_name))
    for file_name in file_names[training_sep: training_sep + dev_sep]:
        shutil.copyfile(os.path.join(midi_dir, file_name), os.path.join(midi_dir, 'dev', file_name))
    for file_name in file_names[training_sep + dev_sep:]:
        shutil.copyfile(os.path.join(midi_dir, file_name), os.path.join(midi_dir, 'test', file_name))


if __name__ == '__main__':
    MIDI_DIR = './../../../datasets/midi/giant_midi_piano'
    separate(MIDI_DIR)
