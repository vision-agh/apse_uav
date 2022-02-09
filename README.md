# Automotive perception system evaluation with reference data from a UAV's camera using ArUco markers and DCNN

The repository contains the code for detecting vehicles and measuring distances between them using ArUCO markers and DCNN as described in [2022 JSPS paper](https://link.springer.com/article/10.1007/s11265-021-01734-3).

<img src="https://user-images.githubusercontent.com/33557183/153049366-af474093-48f1-46b7-96cb-13499a6433ba.gif" width="640" height="360">


# Abstract

Testing and evaluation of an automotive perception system is a complicated task which requires special equipment and infrastructure. To compute key performance indicators and compare the results with real-world situation, some additional sensors and manual data labelling are often required. In this article, we propose a different approach, which is based on a UAV equipped with a 4K camera flying above a test track. Two computer vision methods are used to precisely determine the positions of the objects around the car – one based on ArUco markers and the other on a DCNN (we provide the algorithms used on GitHub). The detections are then correlated with the perception system readings. For the static and dynamic experiments, the differences between various systems are mostly below 0.5 m. The results of the experiments performed indicate that this approach could be an interesting alternative to existing evaluation solutions.

[Full paper](https://link.springer.com/article/10.1007/s11265-021-01734-3)

YouTube video:

[![VIDEO ABSTRACT](http://img.youtube.com/vi/QA7v6006e9w/0.jpg)](http://www.youtube.com/watch?v=QA7v6006e9w)


### Citation

If you use this work in an academic research, please cite the following work:

Blachut, K., Danilowicz, M., Szolc, H. et al. Automotive Perception System Evaluation with Reference Data from a UAV’s Camera Using ArUco Markers and DCNN. J Sign Process Syst (2022). https://doi.org/10.1007/s11265-021-01734-3
```
@Article{Blachut2022JSPS,
  author        = {Krzysztof Blachut and Michal Danilowicz and Hubert Szolc and Mateusz Wasala and Tomasz Kryjak and Mateusz Komorkiewicz},
  title         = {Automotive perception system evaluation with reference data from a UAV's camera using ArUco markers and DCNN},
  journal       = "Journal of Signal Processing Systems",
  year          = 2022,
}
```

# Instruction

### Dependencies

Tested on:
- Ubuntu 18.04
- Python 3.7.4

Required packages:
- opencv-contrib 4.2.0
- numpy 1.17.2
- scipy 1.3.1
- json
- csv

### Datasets

Download static sequence (video or images):
- video https://drive.google.com/file/d/1I5cEaOC-E27OKRbDwrqHVRaCKmC4Z9FO/view?usp=sharing
- images https://drive.google.com/file/d/18vVUi_RFJQ6HU684qBSbJ2oh_ovd6td3/view?usp=sharing

Download dynamic sequence (video or images):
- video https://drive.google.com/file/d/1jfkMxhyzLjUzEqeavAT0mwG-t69Qjm-6/view?usp=sharing
- images https://drive.google.com/file/d/12mKv636uTKLgntGoegtzn9lswswN4vy8/view?usp=sharing

### Run ArUco

**Remember to add full paths to files/folders at first and tune the parameters/flags in the source file!**
A detailed description of the parameters is at the beginning of the `aruco_detect.py` file.
A file with camera parameters `cam_params.json` is included in the repository.
Files with results from DCNN are included in the repository for both static `static_dcnn_data.csv` and dynamic `dynamic_dcnn_data.csv` sequences.
Run the algorithm by typing `python aruco_detect.py`.

### Run DCNN

Instructions for running the fine-tuned network used in the project are [here](../main/dcnn). The video sequences for DCNN were pre-processed (camera distortion removal + gamma correction) to allow a direct comparison with the ArUco method.

Download static sequence for DCNN (video or images):
- video https://drive.google.com/file/d/1qPlXpMykUaxojSXWYx3lk85khxWsAkC4/view?usp=sharing
- images https://drive.google.com/file/d/1-DTcIC3kClF02xTX0xifyumBlqOCh_iQ/view?usp=sharing

Download dynamic sequence for DCNN (video or images):
- video https://drive.google.com/file/d/1A49k4Zwuy2cRhP4bDaA0dTTugeeB7x51/view?usp=sharing
- images https://drive.google.com/file/d/1L8qIxgX7QfPnYqmRClZF2XZmrho7PjCB/view?usp=sharing

### Versions

31.01.2022 - Version 1.0

If you have any questions about the application, feel free to contact us.
