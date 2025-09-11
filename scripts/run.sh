CUDA_VISIBLE_DEVICES=0 bash base2new_train.sh imagenet 0.8 0.0005 3 vit_b16_ep25_ctxv1
CUDA_VISIBLE_DEVICES=0 bash base2new_test.sh imagenet 0.8 0.0005 3 vit_b16_ep25_ctxv1 25

CUDA_VISIBLE_DEVICES=0 bash base2new_train.sh caltech101 0.8 0.0005 3 vit_b16_ep100_ctxv1
CUDA_VISIBLE_DEVICES=0 bash base2new_test.sh caltech101 0.8 0.0005 3 vit_b16_ep100_ctxv1 100

CUDA_VISIBLE_DEVICES=0 bash base2new_train.sh oxford_pets 0.8 0.5 3 vit_b16_ep25_ctxv1
CUDA_VISIBLE_DEVICES=0 bash base2new_test.sh oxford_pets 0.8 0.5 3 vit_b16_ep25_ctxv1 25

CUDA_VISIBLE_DEVICES=0 bash base2new_train.sh stanford_cars 0.8 0.1 3 vit_b16_ep100_ctxv1
CUDA_VISIBLE_DEVICES=0 bash base2new_test.sh stanford_cars 0.8 0.1 3 vit_b16_ep100_ctxv1 100

CUDA_VISIBLE_DEVICES=0 bash base2new_train.sh oxford_flowers 0.8 0.1 3 vit_b16_ep100_ctxv1
CUDA_VISIBLE_DEVICES=0 bash base2new_test.sh oxford_flowers 0.8 0.1 3 vit_b16_ep100_ctxv1 100

CUDA_VISIBLE_DEVICES=0 bash base2new_train.sh food101 0.5 0.1 2 vit_b16_ep25_ctxv1
CUDA_VISIBLE_DEVICES=0 bash base2new_test.sh food101 0.5 0.1 2 vit_b16_ep25_ctxv1 25

CUDA_VISIBLE_DEVICES=0 bash base2new_train.sh fgvc_aircraft 0.8 0.1 2 vit_b16_ep25_ctxv1
CUDA_VISIBLE_DEVICES=0 bash base2new_test.sh fgvc_aircraft 0.8 0.1 2 vit_b16_ep25_ctxv1 25

CUDA_VISIBLE_DEVICES=0 bash base2new_train.sh sun397 0.8 0.1 2 vit_b16_ep25_ctxv1
CUDA_VISIBLE_DEVICES=0 bash base2new_test.sh sun397 0.8 0.1 2 vit_b16_ep25_ctxv1 25

CUDA_VISIBLE_DEVICES=0 bash base2new_train.sh dtd 0.8 0.1 2 vit_b16_ep25_ctxv1
CUDA_VISIBLE_DEVICES=0 bash base2new_test.sh dtd 0.8 0.1 2 vit_b16_ep25_ctxv1 25

CUDA_VISIBLE_DEVICES=0 bash base2new_train.sh eurosat 0.5 0.1 2 vit_b16_ep100_ctxv1
CUDA_VISIBLE_DEVICES=0 bash base2new_test.sh eurosat 0.5 0.1 2 vit_b16_ep100_ctxv1 100

CUDA_VISIBLE_DEVICES=0 bash base2new_train.sh ucf101 0.5 0.1 2 vit_b16_ep25_ctxv1
CUDA_VISIBLE_DEVICES=0 bash base2new_test.sh ucf101 0.5 0.1 2 vit_b16_ep25_ctxv1 25


