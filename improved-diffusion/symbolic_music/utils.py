from miditok import MIDILike, REMI, Structured


def is_midi_task(args):
    return args.modality == 'midi' or args.modality.startswith('midi-')


def get_tokenizer(data_args):
    cls = MIDILike
    if data_args.midi_tokenizer == 'REMI':
        cls = REMI
    elif data_args.midi_tokenizer == 'Structured':
        cls = Structured
    print(f'Use tokenizer {cls.__name__}')
    if data_args.padding_mode == 'bar_block':
        assert data_args.midi_tokenizer == 'REMI'
        return cls(sos_eos_tokens=False, mask=False)
    return cls(sos_eos_tokens=True, mask=False)
