#--------------------------------------------
# base params:
# python main.py --dataname TADPOLE --mode local-global --marl_mode 2  --MP_mode GCN  --n_head 3 --n_hidden 21 --GC_mode weighted-cosine \
#         --reg 0.26150702874057236 --lr 0.007325907184248944 --dropout 0.005331714042633606 \
#         --CL_weight 0.5085609198005367 --CL_rate 0.18777487575294727  --CL_mse 1.1348030844244705\
#         --nlayer 2 --gl_attn_layer 0 --gl_lin_layer 1 \
#         --CL_mode random_mask_edge --seed 27750 \
# acc:92.81
#--------------------------------------------

python main.py --dataname TADPOLE --mode local-global --marl_mode 2  --MP_mode GCN  --n_head 3 --n_hidden 21 --GC_mode weighted-cosine \
         --reg 0.26150702874057236 --lr 0.007325907184248944 --dropout 0.005331714042633606 \
         --nlayer 2 --gl_attn_layer 0 --gl_lin_layer 1 \
         --CL_weight 0.5085609198005367 --CL_rate 0.18777487575294727  --CL_mse 1.1348030844244705 \
         --CL_mode random_mask_edge --seed 27750 \




