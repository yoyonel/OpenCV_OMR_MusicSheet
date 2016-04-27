<p>Thanks for yours informations/codes Abid Rahman K &amp;
bjoernz.
I have re-implemented yours techniques, brute force and morphological operations ("optimized" in term of GPU/CPU usages).</p>

<p>Here some results and timing:
<a href="http://i.stack.imgur.com/afXRq.png" rel="nofollow"><img src="afXRq.png" alt="enter image description here"></a>
Brute force method (from Abid) => 3.05s
<a href="http://i.stack.imgur.com/pGU7P.png" rel="nofollow"><img src="pGU7P.png" alt="enter image description here"></a>
Morpho Operation (from bjoernz) => 0.074s</p>

<p>Results from both techniques seems to be very close but timings are very different ^^ the scale factor close to ~41x (GPU rock for pixels manipulation ^^).</p>

<p>My computer settings: </p>

<ul>
<li>CPU: Intel(R) Core(TM) i5-4440 CPU @ 3.10GHz</li>
<li>GPU: NVidia GeForce GTX 750Ti</li>
<li>(OS: Linux/Mint17 - OpenCV 3.1.0)</li>
</ul>
