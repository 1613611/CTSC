# CTSC

<h2> Args input </h2>

- test-site
  - 4x1-one-way
  - 4x1-two-way
  - 4x2-intersections
- ligh-traffic or heavy-traffic
- gui (optional)
- trial or train (for RL method)
- step-size
- number-episodes-train
- number-episodes-pretrain
- random-seed
- memory-length
- batch-size
- epsilon
- update-interval
- epochs
- gamma
- max-step

<h2> Some examples </h2>
<h3> Fixed-time control </h3>

>python FT.py --test-site 4x1-two-way --light-traffic --gui

Where:
- **--test-site test_site** to select the test site
- **--light-traffic** or **--heavy-traffic** to select the kind of traffic, the default traffic is light traffic
- **--gui** to enable GUI


<h3> GreenWave </h3>

> cd FT_OFFSET </br>
> python FT_OFFSET.py --test-site 4x2-intersections --light-traffic

<h3> SOTL </h3>

> python SOTL.py --test-site 4x1-one-way --max-step 7200 --light-traffic

<h3> MaxPressure </h3>

> python MaxPressure.py --test-site 4x2-intersections --max-step 7200 --light-traffic

<h3> CentrailzedRL </h3>

* To train:

> python CentralizedRL.py  --test-site 4x2-intersections --step-size 5 --number-episodes-train 100 \
>                --number-episodes-pretrain 5 --random-seed 42 --memory-length 4192 --batch-size 512 --epsilon 0.05 \
>                --update-interval 300 --epochs 50 --gamma 0.95 --max-step 7200 --heavy-traffic --train

* To trial:

> python CentralizedRL.py  --test-site 4x2-intersections --step-size 5 --number-episodes-train 100 \
>                --number-episodes-pretrain 5 --random-seed 42 --memory-length 4192 --batch-size 512 --epsilon 0.05 \
>                --update-interval 300 --epochs 50 --gamma 0.95 --max-step 7200 --light-traffic --trial


