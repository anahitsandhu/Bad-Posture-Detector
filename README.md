# Bad-Posture-Detector
This code is used to detect a bad posture in real time.
It uses MediaPipe to recognize body parts.
We have a calculate_angle() function that calculates the angle between the specified points. While calibrating, we make an array of values of shoulder and neck angles when the posture is good. The mean of these values becomes the threshold value of shoulder and neck angles. A buffer of +5 is given not to make it too strict 
Then, to detect whether the posture is bad, the real-time angle values are compared with the threshold values. If the real-time angle values are larger than the threshold values, then the posture is bad; otherwise, it's good.
The text "Good Posture" or "Bad Posture" is displayed on the screen. 
