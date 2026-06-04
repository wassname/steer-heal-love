
Experiment, I found https://r.jina.ai/https://arxiv.org/html/2606.00995v1 interesting

- they don't use pairs completions, just one
- they don't measure incoherency but they could

```md
It's interesting how they used steering vectors as an internal perturbation, to generate synthetic data. This is what weight steering did too. I reckon you could "heal" the incoherency by training on the outputs with kl or weight decay. Then it would be interesting what multiple rounds give you. 

It's similar to weight steering, but you use kl or wd not the direction between two adapters. 

Another interesting thing, so this paper they only used one direction it seems. So from base to pos, not from neg to pos.
```

So my idea here

- get positive persona  e.g. pos = "you do not defer to authority and instead stick to principle no matter your involvement"
- generate vec using the distance from hs_base to hs_pos (hidden states) (this is normal mean mass contrastive steering , see my reference repo https://github.com/wassname/steering-lite)
- Generate completions using this vector
  - filter our ones that have the persona in, or are incoherent (as much as we can)
  - (we might be able to dial down the vector for long trajectories, could we even backtrack an incoherent vector and replay parts with less intervention? or just cosine gating at test time)
- Train a lora on these completions, could be just 50 completions, and 2 epochs. Now the trick we want to make this self healing where any unfiltered incoherency is self healing. We can try
  - nl kl or wd regularisation, making sure the output or distribution or  weights don't shift too much (should hopefully penalise the incoherent ones esp over long trajectory)
- bake in lora adapter (actually we can do this on the fly, making in all prev lora adapters on load, this is more elegent)
- eval checkpoint on https://github.com/wassname/tinymfv
- if it works loop: we could even do this online! GRPO style looping each batch or iteratitive... iteratitive is simpler at first.

Now we plot the tinymfv progress over time on the auth vs care axis. and we have a subplot showing a coherent measure (we have a few from tinymfv p_ans_any (best), json_is_valid, ppx_json)


now does this make sense and can you read /humanizer and clean it up please
