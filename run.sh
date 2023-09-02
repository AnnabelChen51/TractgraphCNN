#python main.py --feature Nos FA1 -outdir 1516 --loss CE --type binary --task sex --net_architecture DGCNN --batch_size 32 --sched_step 100 --epochs 200 --rate 0.000001 --k 20 --seed 3
#python main.py --feature Nos FA1 -outdir 1516 --loss CE --type binary --task sex --net_architecture DGCNN --batch_size 32 --sched_step 100 --epochs 200 --rate 0.000001 --k 20 --seed 3 #test producibility
#python main.py --feature Nos FA1 -outdir 1516 --loss CE --type binary --task sex --net_architecture DGCNN --batch_size 32 --sched_step 100 --epochs 200 --rate 0.000001 --k 20 --seed 1

#python main.py --feature Nos FA1 -outdir 1516 --loss CE --type binary --task sex --net_architecture DGCNN --batch_size 32 --sched_step 100 --epochs 200 --rate 0.000001 --k 20 --seed 1  # add validation set
#python main.py --feature Nos FA1 -outdir 1516--loss CE --type binary --task sex --net_architecture CNN_1D --batch_size 1024 --sched_step 100 --epochs 200 --rate 0.0001 --seed 1

#python main.py --feature Nos FA1 -outdir 1516 --loss CE --type binary --task sex --net_architecture CNN_1D --batch_size 32 --sched_step 100 --epochs 200 --rate 0.0001 --seed 1

#python main.py --feature Nos FA1 -outdir jounral --loss CE --type binary --task sex --net_architecture DGCNN --batch_size 32 --sched_step 100 --epochs 200 --rate 0.000001 --k 20 --seed 3 #953; no Nos_Norm
#python main.py --feature Nos FA1 -outdir jounral --loss CE --type binary --task sex --net_architecture DGCNN --batch_size 32 --sched_step 100 --epochs 200 --rate 0.000001 --k 20 --seed 3 #953; Nos_Norm
#python main.py --feature Nos FA1 -outdir jounral --loss CE --type binary --task sex --net_architecture DGCNN --batch_size 32 --sched_step 100 --epochs 200 --rate 0.000001 --k 20

#python main.py --feature Nos FA1 -outdir val --loss CE --type binary --task sex --net_architecture DGCNN --batch_size 32 --sched_step 100 --epochs 200 --rate 0.000001 --k 20 --seed 3
#python main.py --feature Nos FA1 -outdir val --loss CE --type binary --task sex --net_architecture DGCNN --batch_size 32 --sched_step 100 --epochs 200 --rate 0.000001 --k 20 --seed 1
#python main.py --feature Nos FA1 -outdir val --loss CE --type binary --task sex --net_architecture CNN_1D --batch_size 32 --sched_step 100 --epochs 200 --rate 0.0001 --seed 3
#python main.py --feature Nos FA1 -outdir val --loss CE --type binary --task sex --net_architecture CNN_1D --batch_size 32 --sched_step 100 --epochs 200 --rate 0.0001 --seed 1
#python main.py --feature Nos FA1 -outdir val --loss CE --type binary --task sex --net_architecture DGCNNG --batch_size 16 --sched_step 100 --epochs 200 --rate 0.000001 --seed 3 -dis_file data/dis_sort_roi2.npy
#python main.py --feature Nos FA1 -outdir val --loss CE --type binary --task sex --net_architecture CNN_1D --batch_size 32 --sched_step 100 --epochs 200 --rate 0.0001 --seed 0
#python main.py --feature Nos FA1 -outdir val --loss CE --type binary --task sex --net_architecture DGCNNG --batch_size 16 --sched_step 100 --epochs 200 --rate 0.000001 --seed 0 -dis_file data/dis_sort_roi2.npy

#python main.py --feature Nos FA1 -outdir val --loss CE --type binary --task sex --net_architecture DGCNN --batch_size 32 --sched_step 100 --epochs 200 --rate 0.000001 --k 40 --seed 3
#python main.py --feature Nos FA1 -outdir val --loss CE --type binary --task sex --net_architecture DGCNNG --batch_size 16 --sched_step 100 --epochs 200 --rate 0.000001 --seed 3 -dis_file data/distance_id_sort_1500.npy
#python main.py --feature Nos FA1 -outdir val --loss CE --type binary --task sex --net_architecture DGCNNG --batch_size 16 --sched_step 100 --epochs 200 --rate 0.000001 --seed 3 -dis_file data/distance_id_sort_2000.npy
#
#python main.py --feature Nos FA1 -outdir val --loss CE --type binary --task sex --net_architecture DGCNNG --batch_size 16 --sched_step 100 --epochs 200 --rate 0.000001 --seed 3 -dis_file data/dis_sort_roi2n20.npy
#python main.py --feature Nos FA1 -outdir val --loss CE --type binary --task sex --net_architecture DGCNNG --batch_size 16 --sched_step 100 --epochs 200 --rate 0.000001 --seed 3 -dis_file data/dis_sort_roi2n40.npy
#python main.py --feature Nos FA1 -outdir val --loss CE --type binary --task sex --net_architecture DGCNNG --batch_size 16 --sched_step 100 --epochs 200 --rate 0.000001 --seed 3 -dis_file data/dis_sort_roi2dis1500.npy
#python main.py --feature Nos FA1 -outdir val --loss CE --type binary --task sex --net_architecture DGCNNG --batch_size 16 --sched_step 100 --epochs 200 --rate 0.000001 --seed 3 -dis_file data/dis_sort_roi2dis2000.npy

#transformer
export PROJECT_DIR=/home/yuqian/hdrive/tabular-dl-revisiting-models
#python main.py --feature Nos FA1 -outdir val --loss CE --type binary --task sex --net_architecture TractGraphormer --batch_size 32 --sched_step 100 --epochs 200 --rate 0.000001 --k 20 --seed 3
#python main.py --feature Nos FA1 -outdir val --loss CE --type binary --task sex --net_architecture TractGraphormer --batch_size 32 --sched_step 300 --epochs 200 --rate 0.00001 --k 20 --seed 3
#python main.py --feature Nos FA1 -outdir val --loss CE --type binary --task sex --net_architecture TractGraphormer --batch_size 32 --sched_step 300 --epochs 200 --rate 0.0001 --k 20 --seed 3
#python main.py --feature Nos FA1 -outdir val --loss CE --type binary --task sex --net_architecture TractGraphormer --batch_size 32 --sched_step 300 --epochs 200 --rate 0.000001 --k 20 --seed 1
#python main.py --feature Nos FA1 -outdir val --loss CE --type binary --task sex --net_architecture TractGraphormer --batch_size 32 --sched_step 300 --epochs 200 --rate 0.00001 --k 20 --seed 1
#python main.py --feature Nos FA1 -outdir val --loss CE --type binary --task sex --net_architecture TractGraphormer --batch_size 32 --sched_step 300 --epochs 200 --rate 0.0001 --k 20 --seed 1

#python main.py --feature Nos FA1 -outdir val --loss CE --type binary --task sex --net_architecture TractGraphormer --batch_size 32 --sched_step 300 --epochs 500 --rate 0.000003 --k 20 --seed 1
#python main.py --feature Nos FA1 -outdir val --loss CE --type binary --task sex --net_architecture TractGraphormer --batch_size 32 --sched_step 300 --epochs 500 --rate 0.00003 --k 20 --seed 1
#python main.py --feature Nos FA1 -outdir val --loss CE --type binary --task sex --net_architecture TractGraphormer --batch_size 32 --sched_step 300 --epochs 500 --rate 0.000001 --k 20 --seed 1
#python main.py --feature Nos FA1 -outdir val --loss CE --type binary --task sex --net_architecture TractGraphormer --batch_size 16 --sched_step 300 --epochs 200 --rate 0.00003 --k 20 --seed 1
#python main.py --feature Nos FA1 -outdir val --loss CE --type binary --task sex --net_architecture TractGraphormer --batch_size 8 --sched_step 300 --epochs 200 --rate 0.00003 --k 20 --seed 1
#python main.py --feature Nos FA1 -outdir val --loss CE --type binary --task sex --net_architecture TractGraphormer --batch_size 32 --sched_step 300 --epochs 200 --rate 0.00003 --k 20 --seed 1 #head 4
#python main.py --feature Nos FA1 -outdir val --loss CE --type binary --task sex --net_architecture TractGraphormer --batch_size 32 --sched_step 300 --epochs 200 --rate 0.00003 --k 20 --seed 1 --weight 0.001 #head 1
#python main.py --feature Nos FA1 -outdir val --loss CE --type binary --task sex --net_architecture TractGraphormer --batch_size 32 --sched_step 300 --epochs 200 --rate 0.00003 --k 20 --seed 1 #head 1 # pretrained
#python main.py --feature Nos FA1 -outdir val --loss CE --type binary --task sex --net_architecture TractGraphormer --batch_size 32 --sched_step 300 --epochs 200 --rate 0.00003 --k 20 --seed 1 #head 1
# tranformer without validation set
#python main.py --feature Nos FA1 -outdir nval --loss CE --type binary --task sex --net_architecture DGCNN --batch_size 32 --sched_step 100 --epochs 200 --rate 0.000001 --k 20 --seed 0
#python main.py --feature Nos FA1 -outdir nval --loss CE --type binary --task sex --net_architecture CNN_1D --batch_size 32 --sched_step 100 --epochs 200 --rate 0.0001 --seed 0
#python main.py --feature Nos FA1 -outdir nval --loss CE --type binary --task sex --net_architecture TractGraphormer --batch_size 32 --sched_step 300 --epochs 200 --rate 0.00003 --k 20 --seed 0 # wrong, with validation data
#python main.py --feature Nos FA1 -outdir nval --loss CE --type binary --task sex --net_architecture TractGraphormer --batch_size 32 --sched_step 300 --epochs 200 --rate 0.00003 --k 20 --seed 0
#python main.py --feature Nos FA1 -outdir nval --loss CE --type binary --task sex --net_architecture TractGraphormer --batch_size 32 --sched_step 100 --epochs 200 --rate 0.0001 --k 20 --seed 0 # data seed 1, no val
#python main.py --feature Nos FA1 -outdir nval --loss CE --type binary --task sex --net_architecture TractGraphormer --batch_size 32 --sched_step 100 --epochs 200 --rate 0.0001 --k 20 --seed 0 # data seed 1, val
#python main.py --feature Nos FA1 -outdir nval --loss CE --type binary --task sex --net_architecture TractGraphormer --batch_size 32 --sched_step 100 --epochs 200 --rate 0.0001 --k 20 --seed 0  # data seed 0, no val
#python main.py --feature Nos FA1 -outdir nval --loss CE --type binary --task sex --net_architecture TractGraphormer --batch_size 32 --sched_step 100 --epochs 200 --rate 0.00003 --k 20 --seed 0  # data seed 0, no val
#python main.py --feature Nos FA1 -outdir nval --loss CE --type binary --task sex --net_architecture TractGraphormer --batch_size 32 --sched_step 100 --epochs 200 --rate 0.0001 --k 20 --seed 0 # data seed 0, no val

#final setting
#python main.py --feature Nos FA1 -outdir 0829 --loss CE --type binary --task sex --net_architecture CNN_1D --batch_size 32 --sched_step 100 --epochs 200 --rate 0.0001 --seed 1
#python main.py --feature Nos FA1 -outdir 0829 --loss CE --type binary --task sex --net_architecture DGCNN --batch_size 32 --sched_step 100 --epochs 200 --rate 0.000001 --k 20 --seed 1
#python main.py --feature Nos FA1 -outdir 0829 --loss CE --type binary --task sex --net_architecture TractGraphormer --batch_size 32 --sched_step 300 --epochs 50 --rate 0.00003 --k 20 --seed 1
#python main.py --feature Nos FA1 MD1 -outdir 0829 --loss CE --type binary --task sex --net_architecture TractGraphormer --batch_size 32 --sched_step 300 --epochs 50 --rate 0.00003 --k 20 --seed 1
#python main.py --feature all -outdir 0829 --loss CE --type binary --task sex --net_architecture TractGraphormer --batch_size 32 --sched_step 300 --epochs 50 --rate 0.00003 --k 20 --seed 1
#python main.py --feature Nos FA1 MD1 -outdir 0829 --loss CE --type binary --task sex --net_architecture DGCNN --batch_size 32 --sched_step 100 --epochs 200 --rate 0.000001 --k 20 --seed 1
#python main.py --feature all -outdir 0829 --loss CE --type binary --task sex --net_architecture DGCNN --batch_size 32 --sched_step 100 --epochs 200 --rate 0.000001 --k 20 --seed 1
#python main.py --feature Nos FA1 -outdir 0829 --loss CE --type binary --task sex --net_architecture TractGraphormer --batch_size 32 --sched_step 300 --epochs 120 --rate 0.000001 --k 20 --seed 1
#python main.py --feature Nos FA1 MD1 -outdir 0829 --loss CE --type binary --task sex --net_architecture TractGraphormer --batch_size 32 --sched_step 300 --epochs 120 --rate 0.000001 --k 20 --seed 1
#python main.py --feature Nos FA1 -outdir 0829 --loss CE --type binary --task sex --net_architecture TractGraphormer --batch_size 32 --sched_step 300 --epochs 200 --rate 0.00001 --k 20 --seed 1
#python main.py --feature Nos FA1 MD1 -outdir 0829 --loss CE --type binary --task sex --net_architecture TractGraphormer --batch_size 32 --sched_step 300 --epochs 120 --rate 0.00001 --k 20 --seed 1
#python main.py --feature Nos FA1 -outdir 0829 --loss CE --type binary --task sex --net_architecture TractGraphormer --batch_size 32 --sched_step 300 --epochs 120 --rate 0.00001 --k 20 --seed 1 # no val
#python main.py --feature Nos FA1 -outdir 0829 --loss CE --type binary --task sex --net_architecture TractGraphormer --batch_size 32 --sched_step 300 --epochs 120 --rate 0.00001 --k 20 --seed 1 ## pretrained2
#python main.py --feature Nos FA1 -outdir 0829 --loss CE --type binary --task sex --net_architecture TractGraphormer --batch_size 32 --sched_step 300 --epochs 120 --rate 0.00001 --k 20 --seed 1 # no pretrain
#python main.py --feature Nos FA1 -outdir 0829 --loss CE --type binary --task sex --net_architecture TractGraphormer --batch_size 32 --sched_step 300 --epochs 120 --rate 0.00001 --k 20 --seed 1 # pretrain 1
#python main.py --feature Nos FA1 -outdir 0829 --loss CE --type binary --task sex --net_architecture TractGraphormer --batch_size 32 --sched_step 300 --epochs 120 --rate 0.00001 --k 20 --seed 1 # data aug 0.1
#python main.py --feature Nos FA1 -outdir 0829 --loss CE --type binary --task sex --net_architecture TractGraphormer --batch_size 32 --sched_step 300 --epochs 200 --rate 0.00001 --k 20 --seed 1 # data aug 0.02
#python main.py --feature Nos FA1 -outdir 0829 --loss CE --type binary --task sex --net_architecture DGCNN --batch_size 32 --sched_step 100 --epochs 200 --rate 0.000001 --k 20 --seed 1 # data aug 0.02
#python main.py --feature Nos FA1 -outdir 0829 --loss CE --type binary --task sex --net_architecture DGCNN --batch_size 32 --sched_step 100 --epochs 200 --rate 0.000001 --k 22 --seed 1 # data aug 0.02/22
#python main.py --feature Nos FA1 -outdir 0829 --loss CE --type binary --task sex --net_architecture TractGraphormer --batch_size 32 --sched_step 300 --epochs 200 --rate 0.00001 --k 22 --seed 1 # data aug 0.02/22
#python main.py --feature Nos FA1 -outdir 0829 --loss CE --type binary --task sex --net_architecture TractGraphormer --batch_size 32 --sched_step 300 --epochs 200 --rate 0.00001 --k 20 --seed 1 # data aug 0.02
#python main.py --feature Nos FA1 -outdir 0829 --loss CE --type binary --task sex --net_architecture TractGraphormer --batch_size 32 --sched_step 300 --epochs 200 --rate 0.00001 --k 22 --seed 1 # data aug 22
#python main.py --feature Nos FA1 -outdir 0829 --loss CE --type binary --task sex --net_architecture TractGraphormer --batch_size 32 --sched_step 300 --epochs 200 --rate 0.00001 --k 21 --seed 1 # data aug 0.02/21
#python main.py --feature Nos FA1 -outdir 0829 --loss CE --type binary --task sex --net_architecture CNN_1D --batch_size 32 --sched_step 100 --epochs 200 --rate 0.0001 --seed 0
#python main.py --feature Nos FA1 -outdir 0829 --loss CE --type binary --task sex --net_architecture DGCNN --batch_size 32 --sched_step 100 --epochs 200 --rate 0.000001 --k 20 --seed 1 # data aug 0.02
#python main.py --feature Nos FA1 -outdir 0829 --loss CE --type binary --task sex --net_architecture DGCNN --batch_size 32 --sched_step 100 --epochs 200 --rate 0.000001 --k 20 --seed 1 #data aug 20
#python main.py --feature Nos FA1 -outdir 0829 --loss CE --type binary --task sex --net_architecture TractGraphormer --batch_size 32 --sched_step 300 --epochs 200 --rate 0.00001 --k 20 --seed 1 # pos encode +
#python main.py --feature Nos FA1 -outdir 0829 --loss CE --type binary --task sex --net_architecture TractGraphormer --batch_size 32 --sched_step 300 --epochs 200 --rate 0.00001 --k 20 --seed 1 # pos encode *
#python main.py --feature Nos FA1 -outdir 0829 --loss CE --type binary --task sex --net_architecture TractGraphormer --batch_size 32 --sched_step 300 --epochs 200 --rate 0.00001 --k 20 --seed 1 # ignore
#python main.py --feature Nos FA1 -outdir 0829 --loss CE --type binary --task sex --net_architecture TractGraphormer --batch_size 32 --sched_step 300 --epochs 200 --rate 0.00001 --k 20 --seed 1 --fl 64 # pos encode concatenate
#python main.py --feature Nos FA1 -outdir 0829 --loss CE --type binary --task sex --net_architecture TractGraphormer --batch_size 32 --sched_step 300 --epochs 200 --rate 0.00001 --k 20 --seed 1 --fl 56 # pos encode concatenate
#python main.py --feature Nos FA1 -outdir 0829 --loss CE --type binary --task sex --net_architecture TractGraphormer --batch_size 32 --sched_step 300 --epochs 200 --rate 0.000001 --k 20 --seed 1 # ignore
#python main.py --feature Nos FA1 -outdir 0829 --loss CE --type binary --task sex --net_architecture TractGraphormer --batch_size 32 --sched_step 300 --epochs 200 --rate 0.000001 --k 20 --seed 1 #flatten
#python main.py --feature Nos FA1 -outdir 0829 --loss CE --type binary --task sex --net_architecture TractGraphormer --batch_size 32 --sched_step 300 --epochs 200 --rate 0.00001 --k 20 --seed 1 #flatten
#python main.py --feature Nos FA1 -outdir 0829 --loss CE --type binary --task sex --net_architecture TractGraphormer --batch_size 32 --sched_step 300 --epochs 200 --rate 0.0001 --k 20 --seed 1 #flatten
#python main.py --feature Nos FA1 -outdir 0829 --loss CE --type binary --task sex --net_architecture TractGraphormer --batch_size 32 --sched_step 300 --epochs 200 --rate 0.0001 --k 20 --seed 1 #replace cv compression
#python main.py --feature Nos FA1 -outdir 0829 --loss CE --type binary --task sex --net_architecture TractGraphormer --batch_size 32 --sched_step 300 --epochs 200 --rate 0.00001 --k 20 --seed 1 #replace cv compression
#python main.py --feature Nos FA1 -outdir 0829 --loss CE --type binary --task sex --net_architecture TractGraphormer --batch_size 32 --sched_step 300 --epochs 200 --rate 0.0001 --k 20 --seed 1 #replace cv compression
python main.py --feature Nos FA1 -outdir 0829 --loss CE --type binary --task sex --net_architecture DGCNN --batch_size 32 --sched_step 100 --epochs 200 --rate 0.000001 --k 22 --seed 1 # data aug 22

#parameters to turn: lr,n_head,len_feature