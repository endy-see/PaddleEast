export CUDA_VISIBLE_DEVICES=0,1
python train.py \
--model_save_dir=output/ \
--pretrained_model=../imagenet_resnet50_fusebn.tar.gz \
--data_dir=dataset/coco \
