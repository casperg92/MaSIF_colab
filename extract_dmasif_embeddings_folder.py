from dmasif_embedding_utils import get_model, generate_descr, protonate_pdb, create_folder
import os
import glob
from absl import app
from absl import flags
from data_preprocessing.download_pdb import convert_to_npy

FLAGS = flags.FLAGS


flags.DEFINE_enum('model_resolution', '0.7', ['0.7','1'],
                  'Resolution of the dMaSIF model.')

flags.DEFINE_enum('patch_radius', '12', ['12','9'],
                  'Patch radius of the extracted surface.')

flags.DEFINE_string('input_dir', None,
                    'Input directory to extract dmasif embeddings from.')

flags.mark_flag_as_required('input_dir')

def main(_):
    #parse arguments
    folder_path  = FLAGS.input_dir
    model_resolution = float(FLAGS.model_resolution)
    patch_radius = int(FLAGS.patch_radius) 
    working_dir = '/mnt/data'

    # create folders
    chains_dir = os.path.join(working_dir,'chains')
    create_folder(chains_dir)
    npy_dir = os.path.join(working_dir,'npys')
    create_folder(npy_dir)
    reduce_dir = os.path.join(working_dir,'reduce')
    create_folder(reduce_dir)
    pred_dir = os.path.join(working_dir,'preds')
    create_folder(pred_dir)

    # model parameters 
    model_path, supsampling = get_model(working_dir, model_resolution, patch_radius)

    # Iterate over files inside folder 
    all_files = glob.glob(os.path.join(folder_path, '*.pdb'))
    for target_pdb in all_files: 
        chains = ['A']   #assuming that the protein corresponds to chain A 
        target_name = os.path.splitext(os.path.basename(target_pdb))[0]
        reduced_pdb = protonate_pdb(reduce_dir, target_pdb)
        convert_to_npy(reduced_pdb, chains_dir, npy_dir, chains)
        # Generate the embeddings
        pdb_name = "{n}_{c}_{c}".format(n=target_name, c=chains[0])
        generate_descr(model_path, pred_dir, pdb_name, npy_dir, patch_radius, model_resolution, supsampling)


if __name__ == '__main__':
    app.run(main)