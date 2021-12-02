from snakebids import bids
from snakebids.snakemake_io import glob_wildcards

configfile: 'config.yml'

subjects,tasks = glob_wildcards(config['in_bold'])
train_subjects,train_tasks = glob_wildcards(config['in_bold_train'])
test_subjects,test_tasks = glob_wildcards(config['in_bold_test'])

#print(subjects)
#print(tasks)

#print(train_subjects)
#print(train_tasks)

nvols = 110

imgids = expand(expand('sub-{subject}_task-{task}_vol-{{vol:04d}}',zip,subject=subjects,task=tasks),vol=range(nvols))
train_imgids = expand(expand('sub-{subject}_task-{task}_vol-{{vol:04d}}',zip,subject=train_subjects,task=train_tasks),vol=range(nvols))
test_imgids = expand(expand('sub-{subject}_task-{task}_vol-{{vol:04d}}',zip,subject=test_subjects,task=test_tasks),vol=range(nvols))


#print(imgids)
#print(train_imgids)



rule all_train:
    input:
       expand('trained_models/nnUNet/{arch}/{unettask}/{trainer}__nnUNetPlansv2.1/fold_{fold}/{checkpoint}.model',
                    fold=range(5), 
                    arch=config['architecture'], 
                    unettask=config['unettask'],
                    checkpoint=config['checkpoint'], 
                    trainer=config['trainer'])

rule all_test:
    input:
        predicted_lbls = expand(
                            'raw_data/nnUNet_predictions/{arch}/{unettask}/'
                            '{trainer}__nnUNetPlansv2.1/{checkpoint}/'
                            'fetal_{imgid}_0000.nii.gz', 
                            imgid=test_imgids,
                            arch=config['architecture'],
                            unettask=config['unettask'],
                            checkpoint=config['checkpoint'],
                            trainer=config['trainer'])

rule split:
    """ splits bold 4d vol """
    input: config['in_bold']
    params:
        prefix = bids(subject='{subject}',task='{task}',suffix="bold"),
    output: 
        expand(bids(subject='{subject}',task='{task}',suffix="bold{vol:04d}.nii.gz"),vol=range(nvols),allow_missing=True)
    shell: 'fslsplit {input} {params.prefix} -t'

rule splitmask:
    """ splits 4d mask vol """
    input: config['in_mask']
    params:
        prefix = bids(subject='{subject}',task='{task}',suffix="mask"),
    output: 
        expand(bids(subject='{subject}',task='{task}',suffix="mask{vol:04d}.nii.gz"),vol=range(nvols),allow_missing=True)
    shell: 'fslsplit {input} {params.prefix} -t'

# we resample and zero-pad so the bold data matches the masks.. 
rule resample:
    input: bids(subject='{subject}',task='{task}',suffix="bold{vol}.nii.gz")
    output: bids(subject='{subject}',task='{task}',desc='resampled',suffix="bold{vol}.nii.gz")
    shell: '3dresample -dxyz 3.5 3.5 3.5 -prefix {output} -input {input}'

rule zeropad:
    input: bids(subject='{subject}',task='{task}',desc='resampled',suffix="bold{vol}.nii.gz")
    output: bids(subject='{subject}',task='{task}',desc='zeropad',suffix="bold{vol}.nii.gz")
    shell: '3dZeropad -RL 96 -AP 96 -prefix {output} {input}'


rule cp_training_img:
    input: bids(subject='{subject}',task='{task}',desc='zeropad',suffix="bold{vol}.nii.gz")
    output: 'raw_data/nnUNet_raw_data/{unettask}/imagesTr/fetal_sub-{subject}_task-{task}_vol-{vol}_0000.nii.gz'
    threads: 32 #to make it serial on a node
    group: 'preproc'
    shell: 'cp {input} {output}'

rule cp_test_img:
    input: bids(subject='{subject}',task='{task}',desc='zeropad',suffix="bold{vol}.nii.gz")
    output: 'raw_data/nnUNet_raw_data/{unettask}/imagesTs/fetal_sub-{subject}_task-{task}_vol-{vol}_0000.nii.gz'
    threads: 32 #to make it serial on a node
    group: 'preproc'
    shell: 'cp {input} {output}'




rule cp_training_lbl:
    input: bids(subject='{subject}',task='{task}',suffix="mask{vol}.nii.gz")
    output: 'raw_data/nnUNet_raw_data/{unettask}/labelsTr/fetal_sub-{subject}_task-{task}_vol-{vol}.nii.gz'
    group: 'preproc'
    threads: 32 #to make it serial on a node
    shell: 'cp {input} {output}'



rule create_dataset_json:
    input: 
        training_imgs = expand('raw_data/nnUNet_raw_data/{unettask}/imagesTr/fetal_{imgid}_0000.nii.gz',imgid=train_imgids,  allow_missing=True),
        training_lbls = expand('raw_data/nnUNet_raw_data/{unettask}/labelsTr/fetal_{imgid}.nii.gz',imgid=train_imgids,  allow_missing=True),
        template_json = 'template.json'
    params:
        training_imgs_nosuffix = expand('raw_data/nnUNet_raw_data/{unettask}/imagesTr/fetal_{imgid}.nii.gz',imgid=train_imgids, allow_missing=True),
    output: 
        dataset_json = 'raw_data/nnUNet_raw_data/{unettask}/dataset.json'
    group: 'preproc'
    script: 'create_json.py' 
    
def get_nnunet_env(wildcards):
     return ' && '.join([f'export {key}={val}' for (key,val) in config['nnunet_env'].items()])
 
def get_nnunet_env_tmp(wildcards):
     return ' && '.join([f'export {key}={val}' for (key,val) in config['nnunet_env_tmp'].items()])
 
rule plan_preprocess:
    input: 
        dataset_json = 'raw_data/nnUNet_raw_data/{unettask}/dataset.json'
    params:
        nnunet_env_cmd = get_nnunet_env,
        task_num = lambda wildcards: re.search('Task([0-9]+)\w*',wildcards.unettask).group(1),
    output: 
        dataset_json = 'preprocessed/{unettask}/dataset.json'
    group: 'preproc'
    resources:
        threads = 8,
        mem_mb = 16000
    shell:
        '{params.nnunet_env_cmd} && '
        'nnUNet_plan_and_preprocess  -t {params.task_num} --verify_dataset_integrity'

def get_checkpoint_opt(wildcards, output):
    if os.path.exists(output.latest_model):
        return '--continue_training'
    else:
        return '' 
      
rule train_fold:
    input:
        dataset_json = 'preprocessed/{unettask}/dataset.json',
    params:
        nnunet_env_cmd = get_nnunet_env_tmp,
        rsync_to_tmp = f"rsync -av {config['nnunet_env']['nnUNet_preprocessed']} $SLURM_TMPDIR",
        #add --continue_training option if a checkpoint exists
        checkpoint_opt = get_checkpoint_opt
    output:
#        model_dir = directory('trained_models/nnUNet/{arch}/{unettask}/{trainer}__nnUNetPlansv2.1/fold_{fold}'),
#        final_model = 'trained_models/nnUNet/{arch}/{unettask}/{trainer}__nnUNetPlansv2.1/fold_{fold}/model_final_checkpoint.model',
        latest_model = 'trained_models/nnUNet/{arch}/{unettask}/{trainer}__nnUNetPlansv2.1/fold_{fold}/model_latest.model',
        best_model = 'trained_models/nnUNet/{arch}/{unettask}/{trainer}__nnUNetPlansv2.1/fold_{fold}/model_best.model'
    threads: 16
    resources:
        gpus = 1,
        mem_mb = 64000,
        time = 1440,
    group: 'train'
    shell:
        '{params.nnunet_env_cmd} && '
        '{params.rsync_to_tmp} && '
        'nnUNet_train {params.checkpoint_opt} {wildcards.arch} {wildcards.trainer} {wildcards.unettask} {wildcards.fold}'


rule package_trained_model:
    """ Creates tar file for performing inference with workflow_inference -- note, if you do not run training to completion (1000 epochs), then you will need to clear the snakemake metadata before running this rule, else snakemake will not believe that the model has completed. """
    input:
        latest_model = expand('trained_models/nnUNet/{arch}/{unettask}/{trainer}__nnUNetPlansv2.1/fold_{fold}/{checkpoint}.model',fold=range(5),allow_missing=True),
        latest_model_pkl = expand('trained_models/nnUNet/{arch}/{unettask}/{trainer}__nnUNetPlansv2.1/fold_{fold}/{checkpoint}.model.pkl',fold=range(5),allow_missing=True),
        plan = 'trained_models/nnUNet/{arch}/{unettask}/{trainer}__nnUNetPlansv2.1/plans.pkl'
    params:
        trained_model_dir = config['nnunet_env']['RESULTS_FOLDER'],
        files_to_tar = 'nnUNet/{arch}/{unettask}/{trainer}__nnUNetPlansv2.1'
    output:
        model_tar = 'trained_model.{arch}.{task}.{trainer}.{checkpoint}.tar'
    shell:
        'tar -cvf {output} -C {params.trained_model_dir} {params.files_to_tar}'

rule predict_test_subj:
    input:
        in_training_folder = expand('trained_models/nnUNet/{arch}/{unettask}/{trainer}__nnUNetPlansv2.1/fold_{fold}',fold=range(5),allow_missing=True),
        latest_model = expand('trained_models/nnUNet/{arch}/{unettask}/{trainer}__nnUNetPlansv2.1/fold_{fold}/{checkpoint}.model',fold=range(5),allow_missing=True),
        testing_imgs = expand(
                            'raw_data/nnUNet_raw_data/{unettask}/'
                            'imagesTs/fetal_{imgid}_0000.nii.gz',
                            imgid=test_imgids,
                            allow_missing=True),
    params:
        in_folder = 'raw_data/nnUNet_raw_data/{unettask}/imagesTs',
        out_folder = 'raw_data/nnUNet_predictions/{arch}/{unettask}/{trainer}__nnUNetPlansv2.1/{checkpoint}',
        nnunet_env_cmd = get_nnunet_env,
    output:
        predicted_lbls = expand(
                            'raw_data/nnUNet_predictions/{arch}/{unettask}/'
                            '{trainer}__nnUNetPlansv2.1/{checkpoint}/'
                            'fetal_{imgid}.nii.gz',
                            imgid=test_imgids,
                            allow_missing=True)
    threads: 8 
    resources:
        gpus = 1,
        mem_mb = 32000,
        time = 30,
    group: 'inference'
    shell:
        '{params.nnunet_env_cmd} && '
        'nnUNet_predict  -chk {wildcards.checkpoint}  -i {params.in_folder} -o {params.out_folder} -t {wildcards.unettask}'

   
        

