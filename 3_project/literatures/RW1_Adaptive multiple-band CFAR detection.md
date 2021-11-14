# Adaptive multiple-band CFAR detection

link https://drive.google.com/file/d/1umrHB0QxCTjLPm43422dgQXe-toP-uzT/view

## TL;DR

* algorithm that generalizes CFAR for detecing wether a subimage has object of interest or pure clutter
* Uses Maximum likelihood Estimator

## Summary

* clutters are modeled as Gaussian processes with space-varying Mean. Covariance is assumed to be static or slowly varying
* The latter is not true in practice, if consider the image as a whole. But it can be true if break image into smaller subimages.
* image has $N$ pixels
* image $\textbf{X} : (m,N)$ matrix, intensity at each N pixel is measured in m signal-plus-noise bands. Usually m<=12.
* clutter model: $\bar X = \frac{1}{w^2}[X \circledast W ]$, where $W \in \R^{w,w}$ is all-one kernel of which size has to be chosen to minimze 3rd moment.
* $\textbf{s} = [s(1),...,s(N)]^T$ array of N-column vectors of known signal pattern
* $\textbf{b} = [b(1),...,b(m)]^T$ array of m-column vectors of unknown signal intensities
* residual (removal of clutter): $X_0 = X - \bar X$, $s_0 = s - \bar s$
* The detection algorithms test hypotesis, on $\begin{aligned} H_0&: X = X_0 \text{ (clutter only)} \\ H_A&: X = X_0 + bs_0^T \text{ (clutter plus signal)}\end{aligned}$
* Generalzied likelihood-ratio (GLR) test on each subimage q,
  * test statistic $r(q) = \frac{c_q^T A_q^{-1}c_q}{\alpha_q} \\ \text{ where, } \textbf{c}_q = X_0^{(q)}s_0^{(q)} \in \R^m \\ \textbf{A}_q = X_0^{(q)T}X_0^{(q)} \in \R^{m * m}, \\ \alpha_q = s_0^{(q)T}s_0^{(q)} $
  * for each subimage q, if $r_q \geq r_0$ then $H_A$ else $H_0$. To know $r_0$ detection threshold, we meeed PDF of r(q) given $H_0$.
  * "Probability of a False Alarm (PFA)": the probability that r(q) lies within the rejection region, is the CFAR.
  * $PFA= \int^1_{r_0} f(r|H_0) dr \\ \text{ where } f(r|H_0) = \frac{\Gamma(\frac{n}{2})}{\Gamma(\frac{n-m}{2})\Gamma(\frac{m}{2})}(1-r)^{(n-m-2)/2}r^{(m-2)/2} $
  * normally PFA is predetermined. Closed interval [$r_0$, 1] is the rejection region under $H_0$
  * Hyp Test: If r(q) < $r_0$, we accept $H_0$ with (1-PFA) certainty, otherwise reject $H_0$ and accept $H_A$

## Terminologies

### CFAR (RADAR)

refs:

* https://en.wikipedia.org/wiki/Constant_false_alarm_rate
* https://www.jku.at/fileadmin/gruppen/183/Docs/Finished_Theses/Bachelor_Thesis_Katzlberger_final.pdf

In the radar receiver, the returning echoes are typically received by the antenna, amplified, down-converted to an intermediate frequency, and then passed through detector circuitry that extracts the envelope of the signal, known as the video signal. This video signal is proportional to the power of the received echo and comprises the desired echo signal as well as the unwanted signals from internal receiver noise and external clutter and interference. The term video refers to the resulting signal being appropriate for display on a cathode ray tube, or "video screen".

The role of the **constant false alarm rate** circuitry is to determine the power threshold **above which any return can be considered to probably originate from a target** as opposed to one of the spurious sources.

### CFAR (OPTICs)

refs:

* https://core.ac.uk/download/pdf/82170779.pdf (very easy to read!!!!)

The constant-false-alarm-rate (CFAR) detection algorithm is used for the detection of an optical target in an image dominated by optical clutter. The algorithm can be used for many aerial images when a clutter-subtraction technique is incorporated. To approximate the assumption of a constant covariance matrix, the digital image scene is partitioned into subimages and the CFAR algorithm is applied to the subimages. For each subimage, local means must be computed; the “best” local mean is the one that minimizes the third moment.

## Interesting Projects

* https://github.com/nasir6/py_cfar (cpp)
* https://github.com/iroek/CFAR (py)
