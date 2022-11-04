def is_midi_task(args):
    return args.modality == 'midi' or args.modality.startswith('midi-')
