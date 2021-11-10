# Important Ideas from "Adaptive multiple-band CFAR detection.." 
link https://drive.google.com/file/d/1umrHB0QxCTjLPm43422dgQXe-toP-uzT/view 

## Summary

* generalized CFAR algorithm for detecting presence of dim optical signal of nonzero intensity in a collection of correlated subimage scenes (with signal+noise channels)


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

