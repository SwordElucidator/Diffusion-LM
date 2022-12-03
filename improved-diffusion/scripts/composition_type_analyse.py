# find the most important keywords
import os
from collections import defaultdict, Counter

from miditoolkit import MidiFile


def analyse_giant_midi():
    path = './../../datasets/midi/giant_midi_piano/train'
    names = []
    for midi_file_name in os.listdir(os.path.join(path)):
        # Abt, Franz, 7 Lieder aus dem Buche der Liebe, Op.39, K33a_r6IKeA.mid
        names.extend([j.strip() for i in midi_file_name.split(',') for j in i.split(' ')])
    c = Counter(names)
    import pdb
    pdb.set_trace()


if __name__ == '__main__':
    analyse_giant_midi()


# data augmentation


# 减少层数 减少embedding  √


# evaluation


# video


# paper
