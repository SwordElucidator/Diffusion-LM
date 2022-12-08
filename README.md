# Diffusion-LM on Symbolic Music Generation with Controllability

Based on Diffusion-LM Improves Controllable Text Generation
https://arxiv.org/pdf/2205.14217.pdf 



-----------------------------------------------------
## Conda Setup:
```python 
conda install mpi4py
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install -e improved-diffusion/ 
pip install -e transformers/
pip install spacy==3.2.4 datasets==2.4.0 huggingface_hub==0.4.0 wandb pillow miditok==1.2.9 mpi4py==3.0.3 scipy==1.7.3 miditoolkit==0.1.16
```

-----------------------------------------------------
## Datasets:
https://drive.google.com/file/d/1lmMu8lPo7rsgCIO0yjceap8TUfM72aFK/view?usp=sharing

-----------------------------------------------------
## Train Diffusion-LM:

```cd improved-diffusion; mkdir diffusion_models;```

```python scripts/run_train.py --diff_steps 2000 --model_arch transformer --lr 0.0001 --save_interval 20000 --lr_anneal_steps 200000 --seed 102 --noise_schedule sqrt --in_channel 16 --modality midi --submit no --padding_mode pad --app "--predict_xstart True --training_mode e2e --vocab_size 220 --e2e_train ../datasets/midi/midi_files " --notes xstart_midi --dataset_partition 1 --image_size 16 --midi_tokenizer='REMI'```

```python scripts/run_train.py --diff_steps 2000 --model_arch transformer --lr 0.0001 --save_interval 8000 --lr_anneal_steps 80000 --seed 102 --noise_schedule sqrt --in_channel 16 --modality midi --submit no --padding_mode block --app "--predict_xstart True --training_mode e2e --vocab_size 275 --e2e_train ../datasets/midi/giant_midi_piano " --notes xstart_midi --dataset_partition 1 --image_size 16```


-------------------
## Decode Diffusion-LM:
mkdir generation_outputs 

``python scripts/text_sample.py --model_path diffusion_models/diff_midi_pad_rand16_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd102_xstart_midi/model200000.pt --batch_size 32 --num_samples 32 --top_p 1.0 --out_dir genout1``


------------------- 
## Classifier

### FC
Follow `notebooks/MusicClassifier copy.ipynb`

### Pre-train TransformerNet Model
`` python music_classifier/trainer.py --task=train --data_path=../datasets/midi/giant_midi_piano --experiment=composition_type --outputnohup python music_classifier/trainer.py  --task=pretrain --data_path=../datasets/midi/giant_midi_piano --experiment=composition_type --output_path=./classifier_models/pretrain/ --from_state_path=./classifier_models/pretrain/checkpoint-60000/pytorch_model.bin --from_check_point=./classifier_models/pretrain/checkpoint-60000 --epoches=60 --path_learned=./diffusion_models/diff_midi_giant_midi_piano_REMI_bar_block_rand32_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd102_xstart_midi/model400000.pt``

### TransformerNet Model Fine-Tune

``python music_classifier/trainer.py --task=train --data_path=../datasets/midi/giant_midi_piano --experiment=composition_type --outputnohup python music_classifier/trainer.py --task=train --data_path=../datasets/midi/giant_midi_piano --experiment=composition_type --output_path=./classifier_models/finetune/ --epoches=30 --pretrained_model_path=./classifier_models/pretrain/checkpoint-110000/pytorch_model.bin --path_learned=./diffusion_models/diff_midi_giant_midi_piano_REMI_bar_block_rand32_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd102_xstart_midi/model400000.pt``

------------------- 
## Controllable Midi Generation

### Infill & Length
``python symbolic_music/scripts/infill_length.py --model_path diffusion_models/diff_midi_midi_files_REMI_bar_block_rand32_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd102_xstart_midi/model200000.pt --eval_task_ length --tgt_len 230 --use_ddim True --eta 1. --batch_size 16 --num_samples 16 --out_dir genout_control``

### Classifier Guided Generation

``python symbolic_music/scripts/control_attribute.py --model_path diffusion_models/diff_midi_midi_files_REMI_bar_block_rand32_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd102_xstart_midi/model200000.pt --eval_task_ control_attribute --tgt_len 230 --use_ddim True --eta 1. --batch_size 16 --num_samples 16 --out_dir genout_control``

[//]: # (First, train the classsifier used to guide the generation &#40;e.g. a syntactic parser&#41; )

[//]: # ()
[//]: # (``  )

[//]: # (python train_run.py --experiment e2e-tgt-tree  --app "--init_emb {path-to-diffusion-lm} --n_embd {16} --learned_emb yes " --pretrained_model bert-base-uncased --epoch 6 --bsz 10)

[//]: # (``)

[//]: # ()
[//]: # (Then, we can use the trained classifier to guide generation. )

[//]: # (&#40;currently, need to update the classifier directory in scripts/infill.py. I will clean this up in the next release.&#41;)

[//]: # ()
[//]: # (``python )

[//]: # (python scripts/infill.py --model_path {path-to-diffusion-lm} --eval_task_ 'control_tree' --use_ddim True  --notes "tree_adagrad" --eta 1. --verbose pipe``)



-----------------------------------------------------

```bibtex
@article{Sun-Ouyang-2022-DiffusionLM-symbolic,
  title={Diffusion-LM on Symbolic Music Generation with Controllability},
  author={Hao Sun and Liwen Ouyang},
  year={2022},
}
```
Please also refer to the original paper. 


```bibtex
@article{Li-2022-DiffusionLM,
  title={Diffusion-LM Improves Controllable Text Generation},
  author={Xiang Lisa Li and John Thickstun and Ishaan Gulrajani and Percy Liang and Tatsunori Hashimoto},
  journal={ArXiv},
  year={2022},
  volume={abs/2205.14217}
}
```
### Tokenizers
```bibtex
@inproceedings{miditok2021,
    title={MidiTok: A Python package for MIDI file tokenization},
    author={Nathan Fradet, Jean-Pierre Briot, Fabien Chhel, Amal El Fallah Seghrouchni, Nicolas Gutowski},
    booktitle={Extended Abstracts for the Late-Breaking Demo Session of the 22nd International Society for Music Information Retrieval Conference},
    year={2021}
}
@article{midilike2018,
    title={This time with feeling: Learning expressive musical performance},
    author={Oore, Sageev and Simon, Ian and Dieleman, Sander and Eck, Douglas and Simonyan, Karen},
    journal={Neural Computing and Applications},
    year={2018},
    publisher={Springer}
}
@inproceedings{remi2020,
    title={Pop Music Transformer: Beat-based modeling and generation of expressive Pop piano compositions},
    author={Huang, Yu-Siang and Yang, Yi-Hsuan},
    booktitle={Proceedings of the 28th ACM International Conference on Multimedia},
    year={2020}
}
@inproceedings{cpword2021,
    title={Compound Word Transformer: Learning to Compose Full-Song Music over Dynamic Directed Hypergraphs},
    author={Hsiao, Wen-Yi and Liu, Jen-Yu and Yeh, Yin-Cheng and Yang, Yi-Hsuan},
    booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
    year={2021}
}
```