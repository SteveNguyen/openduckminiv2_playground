# TODO

- [X] add kp randomization
- [] cleanup
- [X] port the reference motion generation here (was ported in another repo)
- [] Understand why it doesn't learn to rotate properly
- [] Relax the stand still penalty ? sometimes it doesn't even try to catch itself
- [X] fit polynoms to the reference motions 

## Ideas and notes

 - Using the polynoms instead of the direct reference motions prevents us from using non cyclical signal such as the body position, which was previously used instead of a linear velocity tracking reward. So I reactivated the velocity tracking.
- Maybe the reference gait frequency is not realistic ? but we already had pretty good results with the same reference motion frequency
- I increased the reference base height by 1cm. Maybe go back to 0.2 instead of 0.1 ?
- Need to threshold the feet contact reference
- maybe the imitation reward is too large and overweighs everything else ?
  - I noticed the catching up when pushed was worst than before
    - but maybe it's because I scaled down the dof vel by 0.05 for sim2real
    - And I added more noise and randomization too
- Maybe the standing still cost also degrades the global performance ?
  - The best walk we had was with the normal reference motion + the exact same reward as disney, where it was pacing on the spot




