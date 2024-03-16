import matplotlib.pyplot as plt
# import IPython.display as ipd

import os
import json
import math
import torch
import torchaudio
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import numpy as np

import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence

from scipy.io.wavfile import write
def main():
    hps = utils.get_hparams_from_file("./configs/later_singer_base.json")

    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model).cuda()
    _ = net_g.eval()
    _ = utils.load_checkpoint("./logs/later_singer_base/G_50000.pth", net_g, None)

    dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps.data)

    with (torch.no_grad()):
        #得到.wav文件
        spec, wav = dataset.get_audio("singer_data/149_林志玲/lzl0246.wav")
        x, x_lengths,y, y_lengths = None,None,None,None
        spec=spec.unsqueeze(0)
        print(spec.size())

        sid_src=torch.LongTensor([11]).cuda()
        spec_lengths=torch.tensor([spec.size(2)]).cuda()
        #spec_lengths=torch.LongTensor([int(spec_lengths)]).cuda()

        sid_tgt1 = torch.LongTensor([2]).cuda()

        net_g = net_g.cuda()
        spec = spec.cuda()

        print("spec shape:", spec.shape)
        print("spec_lengths:", spec_lengths)
        print("sid_src:", sid_src)
        print("sid_tgt1:", sid_tgt1)

        sid_tgt2 = torch.LongTensor([9]).cuda()
        #sid_tgt3 = torch.LongTensor([4]).cuda()
        audio1 = net_g.voice_conversion(spec, spec_lengths, sid_src=sid_src, sid_tgt=sid_tgt1)[0][0,0].data.cpu().float().numpy()
        audio2 = net_g.voice_conversion(spec, spec_lengths, sid_src=sid_src, sid_tgt=sid_tgt2)[0][0,0].data.cpu().float().numpy()
        #audio3 = net_g.voice_conversion(spec, spec_lengths, sid_src=sid_src, sid_tgt=sid_tgt3)[0][0,0].data.cpu().float().numpy()


    output_dir = './out/VC'
    os.makedirs(output_dir, exist_ok=True)

    torchaudio.save(os.path.join(output_dir,'converted_test_1.wav'),torch.tensor(audio1).unsqueeze(0),hps.data.sampling_rate)
    print("Converted audio files saved in", os.path.join(output_dir,'converted_test_1.wav'))
    torchaudio.save(os.path.join(output_dir,'converted_test_2.wav'),torch.tensor(audio2).unsqueeze(0),hps.data.sampling_rate)
    print("Converted audio files saved in", os.path.join(output_dir,'converted_test_2.wav'))
    #torchaudio.save(os.path.join(output_dir,'converted_3.wav'),torch.tensor(audio3),hps.data.sampling_rate)
    #print("Converted audio files saved in", os.path.join(output_dir,'converted_3.wav'))


if __name__ == "__main__":
    main()
