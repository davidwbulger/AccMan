# Create a manim clip showing the creation & use of the FFT.
# (Actually I'm moving toward using this one file for all of the pac man
# accordion video's manim content.)
###############################################################################
""" Run this at command prompt to create high-quality graphics:
manim -qh --fps 30 sceneFFT.py Bicycle
manim -qh --fps 30 sceneFFT.py Heisenberg
manim -tqh --fps 30 sceneFFT.py MMIIL
manim -qh --fps 30 sceneFFT.py Noodle
manim -qh --fps 30 sceneFFT.py ShowFFT3D
manim -qh --fps 30 sceneFFT.py ShowFFT2D
manim -qh --fps 30 sceneFFT.py ShowFFTHaSer
manim -qh --fps 30 sceneFFT.py NN
"""

from manim import *
import numpy as np
import pickle
import scipy.stats

rng = np.random.default_rng()

chl = 2048  #  chunk length in samples
sf = 44100

def getAudioData():
  sampfile = '../training/sample34_20240621.pickle'
  offset = 12345  #  an arbitrary point to start the chunk at
  note = 15  #  Bb, as an example
  with open(sampfile, 'rb') as fid:
    wav = pickle.load(fid)
  wav = wav[note,offset:offset+chl]
  t = np.arange(chl)/sf
  f = np.arange(1,chl//2+1)*sf/chl
  g = np.fft.rfft(wav)[1:]  #  the complex dft
  amp = np.abs(g)
  phase = np.angle(g)
  return (t,wav,f,amp,phase)

def animations_with_start_time(mapping):
  # Gets a single animation that runs many animations, each one with a time
  # delay on start. From u/mathlikeyt at reddit.com/r/manim/comments/17yeda0
  return AnimationGroup(*[AnimationGroup(Wait(t), a, lag_ratio=1)
    for (t,a) in mapping])

# We'll do two separate animations, a 2D one for the time domain & a 3D one
# for the frequency domain.

# manim -pql sceneFFT.py ShowFFT2D
class ShowFFT2D(Scene):
  def construct(self):
    (t,wav,f,amp,phase) = getAudioData()
    T = t[-1]  #  max time, about 0.05 seconds
    ax = Axes(
      x_range=[0,1.1*T,2*T],
      y_range=[*(1.1*np.max(np.abs(wav))*np.array([-1,1])),2],
      x_length=12, y_length=5)
    axlab = ax.get_axis_labels(Text("time").scale(0.7),
      Text("air displacement").scale(0.7))
    alpha = ValueTracker(0)
    waveform = ax.plot_line_graph(t[:2],wav[:2],
      line_color=BLUE, add_vertex_dots=False)
    def wfUp(wf):
      K = int(2 + (chl-2)*alpha.get_value())
      waveform.become(ax.plot_line_graph(t[:K],wav[:K],
        line_color=BLUE, add_vertex_dots=False))
    waveform.add_updater(wfUp)
    self.add(ax, axlab, waveform)
    self.play(alpha.animate.set_value(1),run_time=3)
    self.wait(3)

class NN(Scene):
  # Illustrate the FFT-NN classification chain.
  def construct(self):
    (t,wav,f,amp,phase) = getAudioData()
    T = t[-1]
    ax = Axes(  #  won't be shown
      x_range=[0,T,2*T],
      y_range=[*(np.max(np.abs(wav))*np.array([-1,1])),2],
      x_length=1.8, y_length=1.2).set_x(-5.5)

    def node(lab,fs=28):
      t = Tex(lab,font_size=fs)
      c = Circle(radius=0.6)
      return Group(t,c)

    # The objects on the 'spine' of the illustration:
    waveform = ax.plot_line_graph(t, wav, stroke_width=1,
      line_color=BLUE, add_vertex_dots=False)
    wtofar = Arrow(start=LEFT,end=RIGHT)
    fft = Text('FFT', font_size=36)
    spacer1 = Square()  #  won't be shown
    inlayer = Group(in1 := node(r'$21{1\over2}$ Hz'), in2 := node('43 Hz'),
      in3 := node(r'$64{1\over2}$ Hz'), vd1 := Tex(r'$\vdots$'),
      in9 := node('22 kHz')).arrange(direction=DOWN)
    INs = [in1,in2,in3,in9]  #  input nodes
    spacer2 = Square()  #  won't be shown
    outlayer = Group(on1 := node(r'G$_2$',36), on2 := node(r'A$\flat_2$',36),
      on3 := node(r'A$_2$',36), vd2 := Tex(r'$\vdots$'),
      on9 := node('E$_5$',36)).arrange(direction=DOWN)
    ONs = [on1,on2,on3,on9]  #  output nodes

    flow= Group(waveform,wtofar,fft,spacer1,inlayer,spacer2,outlayer).arrange()

    # Stuff attached to the 'spine':
    wflab = Paragraph("46 ms\naudio\nchunk",font_size=28,alignment='center'
      ).next_to(waveform,UP)
    fftbox = Square().surround(fft)
    arrows1 = [Arrow(start=fftbox.get_right(),end=ink.get_left())
      for ink in INs]
    arrows2 = [[Line(start=inj.get_right(),end=onk.get_left())
      for inj in INs] for onk in ONs]

    self.add(waveform,wflab)
    self.wait(0.5)
    self.play(animations_with_start_time(((0,FadeIn(wtofar)),
      (0.2,FadeIn(fft)),(0.2,FadeIn(fftbox)))))
    self.wait(0.8)
    self.play(animations_with_start_time([ta
      for (ob1,ob2,t) in zip(arrows1,INs,[0,0.3,0.6,1.2])
      for ta in ((t,FadeIn(ob1)),(t,FadeIn(ob2)))] + [(0.9,FadeIn(vd1))]))

    self.wait(1)
    self.play(animations_with_start_time([(t,FadeIn(ob))
      for (arrowSubList,on,too) in zip(arrows2,ONs,[0,4,4.8,6.4])
      for (t,ob) in [(too,on)] +
      [(too+toi,ar) for (ar,toi) in zip(arrowSubList,[0,0.1,0.2,0.3])]] +
      [(5.6,FadeIn(vd2))]))
    self.wait(2)

    [self.play(AnimationGroup(a.animate.set_color(c)
      for asl in arrows2 for a in asl),run_time=0.4) for c in [GRAY,WHITE]*3]
    self.wait(3.6)

class Bicycle(Scene):
  def construct(self):
    circ = Circle(radius=2.8,color='#222222',stroke_width=12)
    PCs = ['C', r'D$\flat$', 'D', r'E$\flat$', 'E', 'F', r'G$\flat$', 'G',
      r'A$\flat$', 'A', r'B$\flat$', 'B']
    theta = ValueTracker(0)

    def notePos(j,th):
      r = 2.8 + 0.2*(-1)**j*(1-np.cos(th))
      phi = (3-j)*PI/6-(j%2)*(th-np.sin(th))/2
      return r*np.array([np.cos(phi),np.sin(phi),0])

    notes = [Tex(pc,color = BLUE if k%3 else RED) for (k,pc) in enumerate(PCs)]
    for (k,n) in enumerate(notes):
      n.add_updater(
        (lambda k:(lambda mob:mob.move_to(notePos(k,theta.get_value()))))(k))

    self.add(circ,*notes)
    self.play(theta.animate.set_value(0),run_time=4)  #  wait should suffice
    self.play(theta.animate.set_value(2*PI),rate_func=rate_functions.linear,
      run_time=3)
    self.wait(4)

class Noodle(Scene):
  def construct(self):
    circ = Circle(radius=2.8,color='#222222',stroke_width=12).move_to(2.8*LEFT)
    PCs = ['C', r'D$\flat$', 'D', r'E$\flat$', 'E', 'F', r'G$\flat$', 'G',
      r'A$\flat$', 'A', r'B$\flat$', 'B']

    def chordPos(k):
      phi = (1-k)*PI/2
      return 1.6*np.array([np.cos(phi)+2.4,np.sin(phi),0])
    def notePos(j):
      phi = (3-j)*PI/6
      return 2.8*np.array([np.cos(phi)-1,np.sin(phi),0])
    notes = [Tex(pc,color = BLUE).set_z_index(1).move_to(notePos(j))
      for (j,pc) in enumerate(PCs)]

    chordSets = (((r'C\,aug',(0,4,8)), (r'E$\flat$aug',(3,7,11)),
      (r'G$\flat$aug',(6,10,2)), (r'A\,aug',(9,1,5))),
      ((r'D$\flat$',(1,5,8)), (r'E$\flat$\,min',(3,6,10)),
      ('G',(7,11,2)), (r'A\,min',(9,0,4))))

    arrows = []  #  collect for easy removal
    chordLabels = []

    idealArrow = Polygon([0,2,0],[2,0,0],[1.3,0,0],[1.3,-2,0],[-1.3,-2,0],
      [-1.3,0,0],[-2,0,0],stroke_width=0,fill_color='#A0A000').set_opacity(1)
    self.add(circ,*notes)
    self.wait(1)
    for chset in chordSets:
      for ob in arrows+chordLabels:
        self.remove(ob)
      for (k,(chname,pclist)) in enumerate(chset):
        newArrows = [idealArrow.copy().move_to(notePos(pc)).rotate(-k*PI/2
          ).scale(.3) for pc in pclist] + [idealArrow.copy().move_to(
          chordPos(k)).rotate(-k*PI/2).scale(.45)]
        newChord = Tex(chname,color=BLACK,font_size=36
          ).set_z_index(1).move_to(chordPos(k))
        self.play(AnimationGroup(FadeIn(newChord),
          *[FadeIn(ar) for ar in newArrows]), run_time=1)
        self.play(AnimationGroup(newChord.animate.set_color(WHITE),
          *[ar.animate.set_opacity(0.4)
          for ar in newArrows]),run_time=0.8)
        arrows += newArrows
        chordLabels.append(newChord)
      self.wait(3)

    self.wait(3)

class Heisenberg(ThreeDScene):
  def construct(self):
    vt = ValueTracker(0)
    NS = 32  #  number of translucent spheres per axes
    aM = 5  #  max abs for visible dots
    aL = 4.4  #  axes' length in each dim
    locaxes = lambda x: always_redraw(lambda:ThreeDAxes(
      x_range=(-aM,aM,2*aM), y_range=(-aM,aM,2*aM), z_range=(-aM,aM,2*aM), 
      x_length=aL, y_length=aL, z_length=aL).set_stroke(width=1,color=GRAY
      ).set_x(x).rotate(0.2*vt.get_value(),UP).rotate(0.2,axis=RIGHT))
      # np.array([1,0,0])))
    axes = [locaxes(x) for x in [-4,4]]

    # Unscaled sphere radii:
    quantiles = 0.8*scipy.stats.norm.ppf((1+(0.5+np.arange(NS))/NS)/2)
    spheres = [[Sphere(center=[x,0,0],resolution=6,fill_color='#4444FF'
      ).set_opacity(0.6/NS).scale_to_fit_width(q) for q in quantiles]
      for x in [-4,4]]

    def scaleClouds(dt):
      for (growShrink,sphSet) in zip([-1,1],spheres):
        factor = np.minimum(2.8,np.exp(growShrink*1.3*np.sin(vt.get_value())))
        for (q,sph) in zip(quantiles,sphSet):
          sph.scale_to_fit_width(q*factor).set_opacity(
            np.minimum(1.0,0.4/NS/factor))

    self.add(Group(Text("Fourier",font_size=36),
      DoubleArrow(start=LEFT,end=RIGHT),
      Text("transform",font_size=36)).arrange(DOWN))
    self.add(Text("Momentum").move_to((4,3.0,0)),
      Text("Position").move_to((-4,3.0,0)))
    self.add(*axes,*(s for sphset in spheres for s in sphset))
    self.add_updater(scaleClouds)
    self.play(vt.animate.set_value(1.5*PI),rate_func=rate_functions.linear,
      run_time=6)

class ShowFFT3D(ThreeDScene):
  def construct(self):
    (t,wav,f,amp,phase) = getAudioData()
    phaseFactor = ValueTracker(1)
    F = f[-1]  #  max freq, maybe 22050Hz
    amp *= 3/np.max(amp) # for visual clarity
    nl = NumberLine(x_range=[0,F,5000], length=11, include_ticks=False,
      include_tip=True)
    nl.add_labels({F:Tex("Frequency")}, direction=np.array([0,0,-1]),
      font_size=48,buff=0.25)
    nl.labels[0].rotate(PI/2,np.array([1,0,0]))
    roots = [nl.n2p(fe) for fe in f]
    R = roots[0]
    # This should totally have worked:
    # phAx = CurvedDoubleArrow(
    #   start_point=np.array([R[0],-1.5*np.sqrt(3),-1.5]),
    #   end_point=np.array([R[0],1.5*np.sqrt(3),-1.5]),
    #   angle=4*PI/3,radius=3,arc_center=R,color=GREEN)
    # But instead I'll do this:
    phAx = CurvedDoubleArrow(
      start_point=np.array([R[0]+np.sqrt(4.5),-np.sqrt(4.5),0]),
      end_point=np.array([R[0]+np.sqrt(4.5),np.sqrt(4.5),0]),
      radius=3,arc_center=R,color=GREEN).rotate(
      -PI/2,np.array([0,1,0]),about_point=R)
    phAxLab = Tex("Phase", color=GREEN).move_to([R[0],0,2.7]).rotate(
      PI/2,np.array([1,0,0])).rotate(-PI/2)
    ampAx = NumberLine(x_range=[0,3.3,5], length=3.3, include_ticks=False,
      include_tip=True).rotate(-PI/2,[0,1,0]).move_to([R[0],0,1.65])
    ampAx.add_labels({3.2:Tex("Amplitude")},font_size=48, buff=0.25,
      direction = np.array([1,0,0]))
    ampAx.labels[0].rotate(PI/2,np.array([1,0,0]))
    lines = [always_redraw(lambda i=i:  #  this i=i thing is mysterious
      Line(roots[i], [roots[i][0],
      amp[i]*np.sin(phaseFactor.get_value()*phase[i]),
      amp[i]*np.cos(phaseFactor.get_value()*phase[i])], color=RED))
      for i in range(len(amp))]

    rate = .7#8  #  return to 1 for full-length video
    self.add(nl,*lines,phAx,phAxLab)  #  ampAx is, I guess, added by FadeIn
    self.set_camera_orientation(PI*5/12, -PI*5/6)
    self.begin_ambient_camera_rotation(PI*rate/15)
    (phi,theta,focDist,gamma,zoom) = self.camera.get_value_trackers()

    self.wait(2/rate)
    self.play(AnimationGroup(FadeOut(phAx), FadeOut(phAxLab), FadeIn(ampAx),
      phaseFactor.animate.set_value(0),
      phi.animate.set_value(PI/2)),run_time=3/rate)
    self.stop_ambient_camera_rotation()
    self.wait(3/rate)

# The next two are extremely similar. I should subclass them to recycle the
# common parts, but I'm too lazy.
class ShowFFTHaSerBb(Scene):
  # Return to the amp vs freq graph from ShowFFT3D to show harmonic series.
  # Some repetition from ShowFFT3D, sorry.
  def construct(self):
    (t,wav,f,amp,phase) = getAudioData()
    F = f[-1]  #  max freq, maybe 22050Hz
    amp *= 3/np.max(amp) # for visual clarity
    ax = Axes(
      x_range=[0,F,2*F],
      y_range=[0,3.3,5],
      x_length=12, y_length=5)
    axlab = ax.get_axis_labels(Text("Frequency").scale(0.7),
      Text("Amplitude").scale(0.7))
    lines = [always_redraw(lambda i=i:  #  this i=i thing is mysterious
      Line(ax.coords_to_point(f[i],0),ax.coords_to_point(f[i],amp[i]),color=RED
      )) for i in range(len(amp))]

    axcop = lambda x,y: ax.coords_to_point(x,y)

    def noteAndOvertones(sym,freq,maxf,vpos,delay):
      texsym = Tex(sym,color=YELLOW).move_to(axcop(freq,vpos))
      cirx = [Circle(color=YELLOW,radius=0.1).move_to(axcop((k+2)*freq,vpos))
        for k in np.arange(maxf/freq-1)]
      return {'anim':animations_with_start_time(
        [(0,FadeIn(texsym))] +
        [(delay+k*0.18, FadeIn(circ)) for (k,circ) in enumerate(cirx)]),
        'mobs':[texsym]+cirx}

    def highlight(xvec,y0,y1,T):
      rex = [RoundedRectangle(corner_radius=0.15,height=1,width=.8,
        fill_color=BLUE_B, fill_opacity=0.3, stroke_width=0).move_to(
        (axcop(x,y0)+axcop(x,y1))/2) for x in xvec]
      return AnimationGroup(
        AnimationGroup(*(FadeIn(rec) for rec in rex),run_time=0.2),
        Wait(T-0.4),
        AnimationGroup(*(FadeOut(rec) for rec in rex),run_time=0.2),
        lag_ratio=1)

    rate = 1  #  return to 1 for full-length video
    zf = 12  #  = zoom factor for the frequency axis
    self.add(ax,axlab,*lines)

    self.wait(1/rate)
    self.play(ax.x_axis.animate.stretch_about_point(zf,0,ax.coords_to_point(
      0,0)), run_time=2/rate)
    self.wait(1/rate)
    self.play(noteAndOvertones(r'B$\flat_3$',233.1,1800,-.2,16/rate)['anim'])
    self.wait(9/rate)

    bflat4 = noteAndOvertones(r'B$\flat_4$',233.1*2,1800,-.5,1/rate)
    self.play(bflat4['anim'])
    self.play(highlight(466.2*np.arange(1,5),-.2,-.5,2/rate))
    self.play(AnimationGroup(FadeOut(mob) for mob in bflat4['mobs']),
      run_time=.5/rate)
    self.wait(1/rate)

    bflat2 = noteAndOvertones(r'B$\flat_2$',233.1/2,1800,-.5,1/rate)
    self.play(bflat2['anim'])
    self.play(highlight(233.1*np.arange(1,9),-.2,-.5,5/rate))
    self.wait(1/rate)

class ShowFFTHaSerFA(Scene):
  # Return to the amp vs freq graph from ShowFFT3D to show harmonic series.
  # Some repetition from ShowFFT3D, sorry.
  def construct(self):
    (t,wav,f,amp,phase) = getAudioData()
    F = f[-1]  #  max freq, maybe 22050Hz
    amp *= 3/np.max(amp) # for visual clarity
    ax = Axes(
      x_range=[0,F,2*F],
      y_range=[0,3.3,5],
      x_length=12, y_length=5)
    axlab = ax.get_axis_labels(Text("Frequency").scale(0.7),
      Text("Amplitude").scale(0.7))
    lines = [always_redraw(lambda i=i:  #  this i=i thing is mysterious
      Line(ax.coords_to_point(f[i],0),ax.coords_to_point(f[i],amp[i]),color=RED
      )) for i in range(len(amp))]

    axcop = lambda x,y: ax.coords_to_point(x,y)

    def noteAndOvertones(sym,freq,maxf,vpos,delay):
      texsym = Tex(sym,color=YELLOW).move_to(axcop(freq,vpos))
      cirx = [Circle(color=YELLOW,radius=0.1).move_to(axcop((k+2)*freq,vpos))
        for k in np.arange(maxf/freq-1)]
      return {'anim':animations_with_start_time(
        [(0,FadeIn(texsym))] +
        [(delay+k*0.18, FadeIn(circ)) for (k,circ) in enumerate(cirx)]),
        'mobs':[texsym]+cirx}

    def highlight(xvec,y0,y1,T):
      rex = [RoundedRectangle(corner_radius=0.15,height=1,width=.8,
        fill_color=BLUE_B, fill_opacity=0.3, stroke_width=0).move_to(
        (axcop(x,y0)+axcop(x,y1))/2) for x in xvec]
      return AnimationGroup(
        AnimationGroup(*(FadeIn(rec) for rec in rex),run_time=0.2),
        Wait(T-0.4),
        AnimationGroup(*(FadeOut(rec) for rec in rex),run_time=0.2),
        lag_ratio=1)

    rate = 1  #  return to 1 for full-length video
    zf = 12  #  = zoom factor for the frequency axis
    self.add(ax,axlab,*lines)

    self.wait(0.1/rate)
    self.play(ax.x_axis.animate.stretch_about_point(zf,0,ax.coords_to_point(
      0,0)), run_time=0.3/rate)
    self.play(noteAndOvertones(r'B$\flat_3$',233.1,1800,-.2,0)['anim'])
    f4 = noteAndOvertones(r'F$_4$',233.1*1.5,1800,-.5,0)
    self.play(f4['anim'])
    self.play(highlight(233.1*3*np.arange(1,3),-.2,-.5,11/rate))
    self.play(AnimationGroup(FadeOut(mob) for mob in f4['mobs']),
      run_time=.5/rate)
    self.wait(1/rate)
    a3 = noteAndOvertones(r'A$_3$',220,1800,-.5,0)
    self.play(a3['anim'])
    self.wait(13/rate)
    self.play(AnimationGroup(FadeOut(mob) for mob in a3['mobs']),
      run_time=.5/rate)
    self.wait(26/rate)

class MMIIL(Scene):
  def construct(self):
    eqns = MathTex(r'2048', r'&= 2^{11}',
      r'\\ &= 2\times 2\times 2\times 2\times 2\times 2\times 2\times ' +
      r'2\times 2\times 2\times 2',color="#FFFFFF",font_size=60)
    self.play(Write(eqns[0],run_time=0.5))
    self.wait(0.5)
    self.play(Write(eqns[1],run_time=0.5))
    self.wait(0.5)
    self.play(Write(eqns[2],run_time=2))
    self.wait(2)
