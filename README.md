# Implementation of Rise Techniques
`Random Masking`: The method involves creating random binary masks that selectively obscure parts of the input image. This process is akin to dimming certain pixels to see how much they contribute to the model’s decision.  
`Model Probing`: These masked images are then fed into the black-box model to observe how the changes affect the output. The model’s response to these perturbations helps determine the importance of different image regions.  
`Importance Calculation`: The importance of each pixel is calculated by averaging the model’s output scores over many random masks, weighted by the presence of the pixel in those masks.\\ Mathematically, this is represented as:

$S(I) = \frac{1}{E[M]} \sum_{i=1}^{N} f(I \odot M_i)M_i$

where $( S(I) )$ is the saliency map, $( E[M] )$ is the expectation of the mask, $( f )$ is the black-box model,$ ( I )$ is the input image, $( \odot )$ denotes element-wise multiplication, and $( M_i )$ are the random masks.
`Monte Carlo Sampling:` The method uses Monte Carlo sampling to empirically estimate the importance map, which allows for a generalizable approach to any black-box model without requiring internal access.