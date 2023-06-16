import argparse
import os.path
import shutil
import tempfile
from subprocess import call

parser = argparse.ArgumentParser('Pajigsaw patch generating script', add_help=False)
parser.add_argument('--output-dir', required=True, type=str)
parser.add_argument('--fragment-dir', required=True, type=str)
args = parser.parse_args()

err_imgs = ['2969c_IRV', '2969c_IRR', '2934c_IRR', '1306b_IRR', '1353f_IRV', '1353f_IRR', '0567o_IRR',
                '1250c_IRR', '1250c_IRV', '1224n_IRV', '1224n_IRR', '1317d_IRV', '1317d_IRR', '2841b_IRR',
                '2841b_IRV', '1445_IRR', '1445_COLR', '1445_IRV', '1445_COLV', '1278c_IRR', '2888e_COLV',
                '2888e_IRV', '0808b_IRV', '0808b_IRR', '0808e_IRV', '0808e_IRR', '1228a_IRR', '1228a_IRV',
                '2911b_IRR', '2911b_IRV', '0808c_IRV', '0808c_IRR', '2969e_IRR', '2969e_IRV', '2915_IRV',
                '2915_IRR', '1322m_IRR', '2996c_IRR', '2996c_IRV', '1290k_IRR', '1290k_IRV', '2930b_IRR',
                '2930b_IRV', '2916c_IRV', '2916c_IRR', '2949c_IRV', '2949c_IRR', '1438o_IRV', '0808a_IRV',
                '0808a_IRR', '2913_IRR', '2913_IRV', '2881d_IRV', '2881d_IRR', '2916b_IRV', '2916b_IRR',
                '1228d_IRV', '1228d_IRR', '1316c_IRR', '1316c_IRV', '1322o_IRV', '1322o_IRR', '2974_IRR',
                '2974_IRV', '2733e_IRR', '2733e_IRV', '2881c_COLR', '2881c_IRR', '2881c_IRV', '2916a_IRV',
                '2916a_IRR', '1306h_IRR', '2855e_COLV', '2855e_IRR', '2855e_COLR', '2838g_COLV', '2949a_IRR',
                '2867f_IRV', '2867f_COLR', '2867f_COLV', '1322k_IRV', '1322k_IRR', '2882b_COLR', '2882b_COLV',
                '2882b_IRR', '0567s_IRV', '2881b_IRV', '0794_IRV', '0794_IRR', '2969b_IRV', '2733r_IRR', '2733r_IRV',
                '2735a_IRV', '2735a_IRR', '1220d_1225e_IRR', '2850b_IRR', '2850b_IRV', '0567r_IRV', '0271p_IRV',
                '0271p_IRR', '2926_IRR', '2926_IRV', '1382i_IRR', '1382i_IRV', '0567o_0567p_0567q_IRV',
                '0567o_0567p_0567q_IRR', '2849d_IRR', '2849d_IRV', '1322l_IRV', '1322l_COLV', '2969d_IRR',
                '2969d_IRV', '1207_1209b_IRV', '2970a_IRR', '2970a_IRV', '2853c_IRR', '2975b_IRR', '2975b_IRV',
                '1223d_IRV', '1290u_IRV', '0779_1257_IRV', '0243_IRV', '0243_IRR', '2855b_COLR',
                '1317c_1321f_1321k_IRR', '2735i_IRR', '2735i_IRV', '1444a_1444b_IRR', '1444a_1444b_IRV',
                '2932_IRR', '2932_IRV', '2952b_IRV', '1224i_IRV', '1224i_IRR', '2949b_IRR', '2882c_COLV',
                '2882c_IRR', '1290r_IRV', '1290r_IRR', '0567n_IRV', '2859a_COLV', '2859a_IRV', '2859a_COLR',
                '2950b_IRV', '1438g_IRR', '1316d_IRV', '1316d_IRR', '2735j_IRR', '2735j_IRV', '1290w_IRV',
                '0811c_1442_IRR', '1223e_1224d_IRV', '1223e_1224d_IRR', '3007_IRR', '3007_IRV', '2970b_IRV',
                '2970b_IRR', '2875f_IRV', '2859b_COLV', '2859b_IRV', '2859b_COLR', '1378i_IRV',
                '0567s_IRR', '0567s_COLR', '0567s_COLV']


def create_temporary_copy(path):
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, os.path.basename(path))
    shutil.copy2(path, temp_path)
    return temp_path


with open('errors.txt') as f:
    error_files = f.readlines()

for err_img in err_imgs:
    identifier, code = err_img.rsplit("_", 1)
    side = code[-1].lower()
    color = code[:-1]
    if color == 'COL':
        color = 'CL'
    image_path = os.path.join(args.fragment_dir, identifier, f'{identifier}_{side}_{color}.JPG')
    assert os.path.exists(image_path)
    error_files.append(image_path)


for file_path in error_files:
    file_path = file_path.strip()
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    identifier, side, color = file_name.rsplit('_', 2)
    file_copy = create_temporary_copy(file_path)
    os.makedirs(os.path.join(args.output_dir, identifier), exist_ok=True)
    if color == 'CL':
        color = 'COL'
    new_name = f"{identifier}_{color}{side.upper()}.jpg"
    out_file = os.path.join(args.output_dir, identifier, new_name)
    if not os.path.exists(out_file):
        call(['shotwell', file_copy])
        shutil.copy2(file_copy, out_file)
    os.remove(file_copy)


