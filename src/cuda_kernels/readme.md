gpu版本，需要关注

亮度模型python层的派生类中 def_shoot(self, photons, rng):

是否有关于 photonArray 关于x, y, flux 的额外操作

已有而外操作的包括：

* [ ] chromatic.py
* [ ] convolve.py
* [X] interpolatedimage.py
* [ ] phase_psf.py
* [ ] sum.py
* [X] transform.py
