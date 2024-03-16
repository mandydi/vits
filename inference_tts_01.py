import argparse
import os
import torch
import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence
from scipy.io.wavfile import write

parser = argparse.ArgumentParser(description='Code to from text to speech')
parser.add_argument('--text', help='the text content to speech', required=True, type=str)
parser.add_argument("--filename", help="filename of output file", required=True,type=str)
parser.add_argument('--checkpoint_path', help='Resumed from this checkpoint', default=None, type=str)
args = parser.parse_args()


if __name__ == "__main__":
    def get_text(text, hps):
        text_norm = text_to_sequence(text, hps.data.text_cleaners)
        if hps.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm


    hps = utils.get_hparams_from_file("./configs/vctk_base.json")

    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model).cuda()
    _ = net_g.eval()

    _ = utils.load_checkpoint(args.checkpoint_path, net_g, None)

    # 保存路径
    output_folder = 'out'
    outpu_filename =args.filename
    output_path=os.path.join(output_folder,outpu_filename)

    # 生成语音
    stn_tst = get_text(args.text, hps)

    with torch.no_grad():
        x_tst = stn_tst.cuda().unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
        sid = torch.LongTensor([4]).cuda()
        audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][
            0, 0].data.cpu().float().numpy()

    #保存生成的语音
    os.makedirs(output_folder,exist_ok=True)
    write(output_path,hps.data.sampling_rate,audio)
    print(f"Save {outpu_filename} sucessfully")