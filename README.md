# scale_adaptive_conv
This repo contains:
1) My naive implementation of conv from https://openaccess.thecvf.com/content_ICCV_2017/papers/Zhang_Scale-Adaptive_Convolutions_for_ICCV_2017_paper.pdf
(adaptive_conv.py, adaModule from adaptive_module.py)
2) My modifications of it
(Tadaptive_conv2.py, adaTrModule from adaptive_module.py)

# If you want test it delete nn.Conv2d in you network and place adaModule instead.
Use with caution adaModule is slow and very memory unfriendly to user module :( But it seems to work.
