# ndlinear-vs-nn.linear
this is to compare the difference in performance between ndlinear and nn.linear by testing it on different data sets

### NDlinear vs nn.linear On Text


![5B09980A-D4D7-42F0-8F7A-3CDA61B12207_4_5005_c](https://github.com/user-attachments/assets/0380d496-3efa-440b-a744-02b1c9c3ec76)


**Text Classification(AG News) Loss Curves**: We trained both a vanilla nn.Linear model and the NdLinear variant for 10 epochs on AG News, using 20‑token inputs, 50‑dim embeddings, and a 32‑dim hidden layer. In epoch 1, both start with a training loss of ~0.69 and a validation loss of ~0.50. By epoch 6, they’ve each dropped to roughly 0.38 train / 0.43 val, with NdLinear already pulling ahead by a few thousandths of a point on the validation set. From epochs 6–10, the curves flatten out—training loss edges down to ~0.37, validation to ~0.426—but NdLinear never relinquishes its lead, ending around 0.426 val versus 0.428 for the baseline. The slightly smaller gap between train and val also suggests NdLinear generalizes just a hair better, all without any extra parameters or slower convergence.


### On UCF‑101 action‐recognition dataset-https://www.crcv.ucf.edu/data/UCF101.php

![AF90BF70-A59B-4393-B7DF-534392D909B3_4_5005_c](https://github.com/user-attachments/assets/945660c3-988c-4cb3-85a5-a4578e2c3848)

***What I did***: I grab a tiny clip of eight 64×64 RGB frames and feed it straight into my NdLinear layer, which—unlike a normal linear layer—knows how to handle the full time×height×width shape without me having to flatten and reshape everything by hand. It spits out a 32‑dim feature vector for each frame. I then average those eight vectors into one summary “video” vector and pass that through a small classification head to predict which of the 101 actions is happening.
***Why I like it:*** Because NdLinear respects the natural spatiotemporal structure, I didn’t have to write any reshaping code, and the model actually learned just a hair faster—hitting about 7 % top‑1 accuracy on UCF‑101 after 10 epochs, compared to around 6.7 % with the vanilla linear version—even though they’re the same size and run in the same time. It’s a tiny tweak in my code that gave me a bit more performance without extra headache.



### Integrating resnet18 to improve performance

![5CC9023B-835D-4B4A-95A7-6FA5AD589121_1_105_c](https://github.com/user-attachments/assets/b146898a-7f65-4a7a-8b91-3c48b9933821)


​​
Both curves build on the **same ResNet‑18** backbone—what you’re seeing is how the two different “heads” (flatten + nn.Linear vs. NdLinear) perform over ten epochs
On the left, validation loss for both models falls rapidly from around 2.9 down toward 2.0 nats, with the NdLinear head (orange) consistently dipping just below the standard head (blue). On the right, you can see validation accuracy climb from roughly 30 % to over 50 %, and again NdLinear nudges ahead by 2–3 percentage points by epoch 10. In short, when you feed identical ResNet features into each head, NdLinear reliably achieves slightly lower loss and slightly higher accuracy without any extra parameters or slowdown.



In my project, I compared a vanilla flatten‑and‑linear head to NdLinear on both text (AG News) and video (UCF‑101) tasks and saw consistent gains. On AG News, both models dropped from ~0.69 train / ~0.50 val loss in epoch 1 to ~0.37 train, ~0.43 val by epoch 6, but NdLinear held a small edge in generalization (0.426 vs. 0.428 val loss) with zero extra parameters. On UCF‑101, raw 8‑frame pixel heads stalled at ~6 % accuracy, but adding a frozen ResNet‑18 backbone boosted both to ~50 % by epoch 10—and NdLinear nudged ahead by 2–3 pp (52.4 % vs. 49.6 % top‑1) and lower loss (2.02 vs. 2.18). The key takeaway is that respecting the data’s natural multi‑D shape yields reliable performance improvements for a one‑line code change, and next I’ll fine‑tune the backbone, scale up capacity, and benchmark speed and memory to fully quantify NdLinear’s benefits.








