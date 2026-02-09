/* ═══════════════════════════════════════════
   concept-probe — GitHub Pages site scripts
   ═══════════════════════════════════════════ */

// ── Helpers ──────────────
const $ = s => document.querySelector(s);
const $$ = s => document.querySelectorAll(s);
const lerp = (a,b,t)=>a+(b-a)*t;
const rand = (a,b)=>Math.random()*(b-a)+a;
const clamp = (v,lo,hi)=>Math.max(lo,Math.min(hi,v));

// roundRect polyfill for older browsers
function safeRoundRect(ctx,x,y,w,h,radii){
  const r = typeof radii === 'number' ? radii : (radii && radii[0]) || 0;
  if(ctx.roundRect){ ctx.beginPath(); ctx.roundRect(x,y,w,h,radii); return; }
  ctx.beginPath();
  ctx.moveTo(x+r,y);
  ctx.lineTo(x+w-r,y); ctx.arcTo(x+w,y,x+w,y+r,r);
  ctx.lineTo(x+w,y+h); ctx.arcTo(x+w,y+h,x+w-r,y+h,0);
  ctx.lineTo(x,y+h); ctx.arcTo(x,y+h,x,y+h-r,0);
  ctx.lineTo(x,y+r); ctx.arcTo(x,y,x+r,y,r);
  ctx.closePath();
}

// ── Mobile nav ──────────
$('#hamburger').addEventListener('click',()=>{
  $('#nav-links').classList.toggle('open');
});
document.addEventListener('click',e=>{
  if(!e.target.closest('nav')) $('#nav-links').classList.remove('open');
});

// ── Scroll reveal ──────────
const revealObs = new IntersectionObserver((entries)=>{
  entries.forEach(e=>{if(e.isIntersecting){e.target.classList.add('visible');revealObs.unobserve(e.target);}});
},{threshold:.12});
$$('.reveal').forEach(el=>revealObs.observe(el));

// ── Copy code button ──────────
window.copyCode = function(btn){
  const pre = btn.closest('.code-block').querySelector('pre');
  navigator.clipboard.writeText(pre.textContent).then(()=>{btn.textContent='Copied!';setTimeout(()=>btn.textContent='Copy',1500);});
};

// ── Stat counters ──────────
const counterObs = new IntersectionObserver((entries)=>{
  entries.forEach(e=>{
    if(!e.isIntersecting) return;
    const el = e.target;
    const target = +el.dataset.target;
    let cur = 0;
    const step = Math.ceil(target/30);
    const iv = setInterval(()=>{cur=Math.min(cur+step,target);el.textContent=cur;if(cur>=target)clearInterval(iv);},40);
    counterObs.unobserve(el);
  });
},{threshold:.5});
$$('.stat-num[data-target]').forEach(el=>counterObs.observe(el));

/* ═══════════════════════════════════════════
   HERO BACKGROUND — floating particles
   ═══════════════════════════════════════════ */
(function(){
  const c = document.getElementById('hero-canvas');
  if(!c) return;
  const ctx = c.getContext('2d');
  let W,H;
  function resize(){W=c.width=c.offsetWidth;H=c.height=c.offsetHeight;}
  resize(); window.addEventListener('resize',resize);

  const N = 60;
  const pts = Array.from({length:N},()=>({x:rand(0,1),y:rand(0,1),vx:rand(-.0003,.0003),vy:rand(-.0003,.0003),r:rand(1,2.5)}));

  function draw(){
    ctx.clearRect(0,0,W,H);
    // connections
    for(let i=0;i<N;i++){
      for(let j=i+1;j<N;j++){
        const dx=(pts[i].x-pts[j].x)*W, dy=(pts[i].y-pts[j].y)*H;
        const d=Math.sqrt(dx*dx+dy*dy);
        if(d<160){
          ctx.beginPath();
          ctx.moveTo(pts[i].x*W,pts[i].y*H);
          ctx.lineTo(pts[j].x*W,pts[j].y*H);
          ctx.strokeStyle=`rgba(99,102,241,${.15*(1-d/160)})`;
          ctx.lineWidth=.6;
          ctx.stroke();
        }
      }
    }
    // dots
    pts.forEach(p=>{
      p.x+=p.vx; p.y+=p.vy;
      if(p.x<0||p.x>1)p.vx*=-1;
      if(p.y<0||p.y>1)p.vy*=-1;
      ctx.beginPath();
      ctx.arc(p.x*W,p.y*H,p.r,0,Math.PI*2);
      ctx.fillStyle='rgba(167,139,250,.5)';
      ctx.fill();
    });
    requestAnimationFrame(draw);
  }
  draw();
})();

/* ═══════════════════════════════════════════
   HERO — Concept Vector Extraction Animation
   ═══════════════════════════════════════════ */
(function(){
  const c = document.getElementById('concept-anim-canvas');
  if(!c) return;
  const ctx = c.getContext('2d');
  const W=c.width, H=c.height;

  // Layers (vertical slices)
  const numLayers = 8;
  const layerGap = W/(numLayers+1);
  const neuronsPerLayer = 6;
  const neuronGap = (H-80)/neuronsPerLayer;
  const startY = 50;

  // Build neuron positions
  const layers = [];
  for(let l=0;l<numLayers;l++){
    const neurons = [];
    const x = layerGap*(l+1);
    for(let n=0;n<neuronsPerLayer;n++){
      neurons.push({x, y:startY+neuronGap*(n+.5), activation:0, phase:rand(0,Math.PI*2)});
    }
    layers.push(neurons);
  }

  // "Best layer" index
  const bestLayer = 5;
  let time = 0;

  function draw(){
    time += 0.015;
    ctx.clearRect(0,0,W,H);

    // Draw connections
    for(let l=0;l<numLayers-1;l++){
      for(const n1 of layers[l]){
        for(const n2 of layers[l+1]){
          ctx.beginPath();
          ctx.moveTo(n1.x,n1.y);
          ctx.lineTo(n2.x,n2.y);
          ctx.strokeStyle='rgba(99,102,241,.06)';
          ctx.lineWidth=.5;
          ctx.stroke();
        }
      }
    }

    // Concept direction arrow at best layer
    const bx = layerGap*(bestLayer+1);
    const arrowPhase = (Math.sin(time*1.5)+1)/2;
    const arrowLen = 60 + arrowPhase*20;
    const arrowY1 = H/2 - arrowLen/2;
    const arrowY2 = H/2 + arrowLen/2;

    ctx.save();
    ctx.shadowColor='rgba(52,211,153,.6)';
    ctx.shadowBlur=16;
    ctx.beginPath();
    ctx.moveTo(bx+30, arrowY1);
    ctx.lineTo(bx+30, arrowY2);
    ctx.strokeStyle='#34d399';
    ctx.lineWidth=2.5;
    ctx.stroke();
    // arrowhead
    ctx.beginPath();
    ctx.moveTo(bx+30, arrowY1-4);
    ctx.lineTo(bx+24, arrowY1+10);
    ctx.lineTo(bx+36, arrowY1+10);
    ctx.closePath();
    ctx.fillStyle='#34d399';
    ctx.fill();
    ctx.restore();

    // Label
    ctx.fillStyle='rgba(52,211,153,.8)';
    ctx.font='bold 10px Inter,sans-serif';
    ctx.textAlign='center';
    ctx.fillText('concept vector', bx+30, arrowY2+16);

    // Draw neurons
    for(let l=0;l<numLayers;l++){
      const isBest = l===bestLayer;
      for(const n of layers[l]){
        n.activation = .3 + .7*(.5+.5*Math.sin(time*2+n.phase+l*.5));
        const glow = isBest ? .6 : .15;
        const r = isBest ? 6 : 4.5;
        const alpha = .3 + n.activation * .7;

        ctx.beginPath();
        ctx.arc(n.x,n.y,r,0,Math.PI*2);
        if(isBest){
          ctx.fillStyle=`rgba(52,211,153,${alpha})`;
          ctx.shadowColor='rgba(52,211,153,.5)';
          ctx.shadowBlur=10;
        } else {
          ctx.fillStyle=`rgba(99,102,241,${alpha*.6})`;
          ctx.shadowColor='rgba(99,102,241,.2)';
          ctx.shadowBlur=4;
        }
        ctx.fill();
        ctx.shadowBlur=0;
      }

      // Layer label
      ctx.fillStyle = l===bestLayer ? 'rgba(52,211,153,.7)' : 'rgba(148,163,184,.35)';
      ctx.font = `${l===bestLayer?'bold ':''}9px Inter,sans-serif`;
      ctx.textAlign='center';
      ctx.fillText(`L${l}`, layers[l][0].x, H-8);
    }

    // Moving data pulses along connections
    for(let l=0;l<numLayers-1;l++){
      const progress = ((time*1.2 + l*0.3) % 2) / 2; // 0-1
      if(progress < 1){
        const n1 = layers[l][Math.floor(neuronsPerLayer/2)];
        const n2 = layers[l+1][Math.floor(neuronsPerLayer/2)];
        const px = lerp(n1.x,n2.x,progress);
        const py = lerp(n1.y,n2.y,progress);
        ctx.beginPath();
        ctx.arc(px,py,2.5,0,Math.PI*2);
        ctx.fillStyle='rgba(34,211,238,.7)';
        ctx.shadowColor='rgba(34,211,238,.5)';
        ctx.shadowBlur=8;
        ctx.fill();
        ctx.shadowBlur=0;
      }
    }

    // Labels
    ctx.fillStyle='rgba(226,232,240,.5)';
    ctx.font='10px Inter,sans-serif';
    ctx.textAlign='left';
    ctx.fillText('Input tokens', 8, 20);
    ctx.textAlign='right';
    ctx.fillText('Output', W-8, 20);

    // Best layer bracket
    if(true){
      ctx.fillStyle='rgba(52,211,153,.5)';
      ctx.font='bold 10px Inter,sans-serif';
      ctx.textAlign='center';
      ctx.fillText('★ best layer', bx, 22);
    }

    requestAnimationFrame(draw);
  }
  draw();
})();

/* ═══════════════════════════════════════════
   PHASE 1 — Generation animation
   ═══════════════════════════════════════════ */
(function(){
  const container = document.getElementById('gen-anim');
  if(!container) return;

  const posTexts = [
    "I feel so tired and numb…",
    "Everything seems pointless today.",
  ];
  const negTexts = [
    "What a wonderful morning!",
    "I'm excited to start the day!",
  ];

  function makeCard(label, cls, text){
    const d = document.createElement('div');
    d.className = 'gen-card '+cls;
    d.innerHTML = `<div class="gen-card-label">${label} system</div><div class="gen-text"></div><span class="gen-cursor"></span>`;
    return d;
  }

  const cards = [];
  posTexts.forEach(t=>{ const c=makeCard('positive','pos',t); container.appendChild(c); cards.push({el:c,text:t,idx:0}); });
  negTexts.forEach(t=>{ const c=makeCard('negative','neg',t); container.appendChild(c); cards.push({el:c,text:t,idx:0}); });

  // Observe then start typing
  const obs = new IntersectionObserver((entries)=>{
    entries.forEach(e=>{
      if(!e.isIntersecting) return;
      obs.unobserve(e.target);
      startTyping();
    });
  },{threshold:.3});
  obs.observe(container);

  let started = false;
  function startTyping(){
    if(started)return; started=true;
    cards.forEach((c,ci)=>{
      const span = c.el.querySelector('.gen-text');
      let i=0;
      const iv = setInterval(()=>{
        if(i<=c.text.length){
          span.textContent = c.text.slice(0,i);
          i++;
        } else {
          clearInterval(iv);
          c.el.querySelector('.gen-cursor').style.display='none';
        }
      }, 35 + ci*8);
    });
  }
})();

/* ═══════════════════════════════════════════
   PHASE 2 — Hidden States Extraction Animation
   ═══════════════════════════════════════════ */
(function(){
  const c = document.getElementById('layers-anim-canvas');
  if(!c) return;
  const ctx = c.getContext('2d');
  let W, H;
  function resize(){ W=c.width=c.offsetWidth; H=c.height=c.offsetHeight; }
  resize(); window.addEventListener('resize',resize);

  const numLayers = 12;
  let time = 0;
  let running = false;

  const obs = new IntersectionObserver((entries)=>{
    entries.forEach(e=>{
      if(e.isIntersecting && !running){ running=true; animate(); }
      if(!e.isIntersecting) running=false;
    });
  },{threshold:.3});
  obs.observe(c);

  function animate(){
    if(!running) return;
    time += 0.02;
    ctx.clearRect(0,0,W,H);

    const gap = (W-60)/numLayers;
    const barW = gap * .65;

    for(let l=0;l<numLayers;l++){
      const x = 30 + l*gap;
      const h = 30 + 80 * (.5+.5*Math.sin(time*1.5 + l*.6));

      // Bar
      const grad = ctx.createLinearGradient(x, H-20-h, x, H-20);
      grad.addColorStop(0,'rgba(99,102,241,.7)');
      grad.addColorStop(1,'rgba(99,102,241,.15)');
      ctx.fillStyle = grad;
      safeRoundRect(ctx,x,H-20-h,barW,h,[4,4,0,0]);
      ctx.fill();

      // Scanning beam
      const scanX = 30 + ((time*2.5)%(numLayers))*gap;
      if(Math.abs(x - scanX) < gap*.6){
        ctx.fillStyle='rgba(34,211,238,.25)';
        ctx.fillRect(x-2, 10, barW+4, H-25);
      }

      // Label
      ctx.fillStyle='rgba(148,163,184,.5)';
      ctx.font='9px JetBrains Mono,monospace';
      ctx.textAlign='center';
      ctx.fillText(`L${l}`, x+barW/2, H-6);
    }

    // Title
    ctx.fillStyle='rgba(226,232,240,.6)';
    ctx.font='bold 11px Inter,sans-serif';
    ctx.textAlign='left';
    ctx.fillText('Hidden states per layer', 10, 18);

    // Scan line
    const scanX = 30 + ((time*2.5)%numLayers)*gap + barW/2;
    ctx.beginPath();
    ctx.moveTo(scanX, 22);
    ctx.lineTo(scanX, H-22);
    ctx.strokeStyle='rgba(34,211,238,.5)';
    ctx.lineWidth=1.5;
    ctx.setLineDash([4,4]);
    ctx.stroke();
    ctx.setLineDash([]);

    requestAnimationFrame(animate);
  }
})();

/* ═══════════════════════════════════════════
   PHASE 3 — Concept Vector Formation
   ═══════════════════════════════════════════ */
(function(){
  const c = document.getElementById('vector-anim-canvas');
  if(!c) return;
  const ctx = c.getContext('2d');
  let W, H;
  function resize(){ W=c.width=c.offsetWidth; H=c.height=c.offsetHeight; }
  resize(); window.addEventListener('resize',resize);

  let time = 0;
  let running = false;

  const obs = new IntersectionObserver((entries)=>{
    entries.forEach(e=>{
      if(e.isIntersecting && !running){ running=true; animate(); }
      if(!e.isIntersecting) running=false;
    });
  },{threshold:.3});
  obs.observe(c);

  function animate(){
    if(!running) return;
    time += 0.015;
    ctx.clearRect(0,0,W,H);

    const cx = W/2, cy = H/2;

    // Pos cluster
    const posX = cx - 80, posY = cy - 10;
    // Neg cluster
    const negX = cx + 80, negY = cy + 10;

    // Draw clusters
    const nPts = 8;
    const posColor='rgba(248,113,113,';
    const negColor='rgba(96,165,250,';

    for(let i=0;i<nPts;i++){
      const ang = (i/nPts)*Math.PI*2 + time*.3;
      const r = 18 + 8*Math.sin(time+i);
      const px = posX + Math.cos(ang)*r;
      const py = posY + Math.sin(ang)*r;
      ctx.beginPath(); ctx.arc(px,py,3.5,0,Math.PI*2);
      ctx.fillStyle=posColor+'.6)'; ctx.fill();

      const nx = negX + Math.cos(ang+Math.PI)*r;
      const ny = negY + Math.sin(ang+Math.PI)*r;
      ctx.beginPath(); ctx.arc(nx,ny,3.5,0,Math.PI*2);
      ctx.fillStyle=negColor+'.6)'; ctx.fill();
    }

    // Mean dots
    ctx.beginPath(); ctx.arc(posX,posY,6,0,Math.PI*2);
    ctx.fillStyle=posColor+'.9)'; ctx.fill();
    ctx.beginPath(); ctx.arc(negX,negY,6,0,Math.PI*2);
    ctx.fillStyle=negColor+'.9)'; ctx.fill();

    // Concept vector arrow (μ_pos → μ_neg direction, normalized)
    const dx = posX-negX, dy = posY-negY;
    const len = Math.sqrt(dx*dx+dy*dy);
    const ux = dx/len, uy = dy/len;
    const vLen = 60 + 10*Math.sin(time*2);
    const ax = cx - ux*vLen/2, ay = cy - uy*vLen/2;
    const bx = cx + ux*vLen/2, by = cy + uy*vLen/2;

    ctx.save();
    ctx.shadowColor='rgba(52,211,153,.6)'; ctx.shadowBlur=12;
    ctx.beginPath(); ctx.moveTo(ax,ay); ctx.lineTo(bx,by);
    ctx.strokeStyle='#34d399'; ctx.lineWidth=3; ctx.stroke();
    // arrowhead at pos end
    const ahs=8;
    ctx.beginPath();
    ctx.moveTo(bx, by);
    ctx.lineTo(bx-ux*ahs-uy*ahs*.6, by-uy*ahs+ux*ahs*.6);
    ctx.lineTo(bx-ux*ahs+uy*ahs*.6, by-uy*ahs-ux*ahs*.6);
    ctx.closePath();
    ctx.fillStyle='#34d399'; ctx.fill();
    ctx.restore();

    // Labels
    ctx.fillStyle='rgba(248,113,113,.8)'; ctx.font='bold 10px Inter,sans-serif'; ctx.textAlign='center';
    ctx.fillText('μ_pos (sad)', posX, posY-30);
    ctx.fillStyle='rgba(96,165,250,.8)';
    ctx.fillText('μ_neg (happy)', negX, negY+40);
    ctx.fillStyle='rgba(52,211,153,.9)'; ctx.font='bold 11px Inter,sans-serif';
    ctx.fillText('v = norm(μ_pos − μ_neg)', cx, 18);

    requestAnimationFrame(animate);
  }
})();

/* ═══════════════════════════════════════════
   LAYER SWEEP — Animated bar chart
   ═══════════════════════════════════════════ */
(function(){
  const container = document.getElementById('sweep-container');
  const bestLabel = document.getElementById('sweep-best-label');
  if(!container) return;

  const numLayers = 32;
  const bestLayer = 22;
  // Generate fake Cohen's d values (peak around layer 22)
  const values = Array.from({length:numLayers},(_,i)=>{
    const dist = Math.abs(i-bestLayer);
    return Math.max(0, 2.4*Math.exp(-dist*dist/40) + rand(-.15,.15));
  });
  const maxVal = Math.max(...values);

  // Build bars
  for(let i=0;i<numLayers;i++){
    const row = document.createElement('div');
    row.className='sweep-bar-row';
    const label = document.createElement('div');
    label.className='sweep-bar-label';
    label.textContent = `L${i}`;
    const bar = document.createElement('div');
    bar.className='sweep-bar' + (i===bestLayer?' best':'');
    const fill = document.createElement('div');
    fill.className='sweep-bar-fill';
    const pct = (values[i]/maxVal)*100;
    const hue = i===bestLayer ? '160,84%,54%' : `${240+i*2},70%,${50+values[i]/maxVal*20}%`;
    fill.style.background = i===bestLayer
      ? 'linear-gradient(90deg,rgba(52,211,153,.7),rgba(52,211,153,.9))'
      : `linear-gradient(90deg,rgba(99,102,241,.3),rgba(99,102,241,.7))`;
    fill.dataset.width = pct+'%';
    bar.appendChild(fill);
    row.appendChild(label);
    row.appendChild(bar);
    container.appendChild(row);

    // Show only every 4th label on small screens
    if(i%4!==0 && i!==bestLayer && i!==numLayers-1) label.style.opacity='.3';
  }

  // Animate on scroll
  const obs = new IntersectionObserver((entries)=>{
    entries.forEach(e=>{
      if(!e.isIntersecting) return;
      obs.unobserve(e.target);
      // Stagger bar fill animation
      const fills = container.querySelectorAll('.sweep-bar-fill');
      fills.forEach((f,i)=>{
        setTimeout(()=>{
          f.style.width = f.dataset.width;
        }, i*40);
      });
      // Show best label after animation
      setTimeout(()=>{
        bestLabel.textContent = `★ Best layer: L${bestLayer} (d=${values[bestLayer].toFixed(2)})`;
        bestLabel.style.opacity='1';
      }, numLayers*40 + 400);
    });
  },{threshold:.2});
  obs.observe(container);
})();

/* ═══════════════════════════════════════════
   TOKEN HEATMAP — Animated demo
   ═══════════════════════════════════════════ */
(function(){
  const container = document.getElementById('heatmap-demo');
  if(!container) return;

  const messages = [
    {role:'system',tokens:['You','are','a','helpful','assistant','.']},
    {role:'user',tokens:['Write','a','short','paragraph','about','the','ocean','.']},
    {role:'assistant',tokens:['The','ocean','stretches','endlessly',',','a','vast','expanse','of','deep','blue','that','hums','with','quiet','melancholy','.','Its','waves','crash','softly',',','carrying','the','weight','of','forgotten','sorrows','to','distant','shores','.']},
  ];

  // Scores — assistant tokens get meaningful values, rest near 0
  const scoreMap = {
    'system': ()=>rand(-.05,.05),
    'user': ()=>rand(-.08,.08),
    'assistant': null, // custom per token
  };
  const assistantScores = [.05,.12,.08,.35,.02,.01,.15,.22,.03,.42,.18,.04,.08,.06,.10,.85,.03,.07,.55,.10,.08,.01,.15,.72,.06,.55,.65,.12,.08,.20,.02];

  messages.forEach(msg=>{
    const row = document.createElement('div');
    row.className='heatmap-row';

    const label = document.createElement('span');
    label.className='hm-label '+msg.role;
    label.textContent=msg.role;
    row.appendChild(label);

    msg.tokens.forEach((tok,ti)=>{
      const span = document.createElement('span');
      span.className='hm-token';
      span.textContent=tok;
      const score = msg.role==='assistant' ? (assistantScores[ti]||rand(-.1,.4)) : scoreMap[msg.role]();
      span.dataset.score=score.toFixed(3);

      const tip = document.createElement('span');
      tip.className='tooltip';
      tip.textContent=score.toFixed(3);
      span.appendChild(tip);

      span.style.opacity='0';
      span.style.transform='translateY(8px)';
      row.appendChild(span);
    });

    container.appendChild(row);
  });

  // Animate on scroll
  const obs = new IntersectionObserver((entries)=>{
    entries.forEach(e=>{
      if(!e.isIntersecting) return;
      obs.unobserve(e.target);
      const tokens = container.querySelectorAll('.hm-token');
      tokens.forEach((tok,i)=>{
        setTimeout(()=>{
          tok.style.opacity='1';
          tok.style.transform='translateY(0)';
          const s = parseFloat(tok.dataset.score);
          if(s>0){
            const intensity = Math.min(1, s/.85);
            tok.style.background=`rgba(239,68,68,${intensity*.5})`;
          } else {
            const intensity = Math.min(1, Math.abs(s)/.85);
            tok.style.background=`rgba(59,130,246,${intensity*.5})`;
          }
        }, i*30);
      });
    });
  },{threshold:.2});
  obs.observe(container);
})();

/* ═══════════════════════════════════════════
   STEERING — Interactive slider demo
   ═══════════════════════════════════════════ */
(function(){
  const slider = document.getElementById('steering-slider');
  const output = document.getElementById('steering-output');
  const label = document.getElementById('alpha-label');
  if(!slider) return;

  const texts = {
    '-8':   "The ocean sparkles with brilliant light today! Every wave seems to dance with pure joy, catching sunbeams and throwing them back as rainbow prisms. I feel an overwhelming sense of gratitude just watching this magnificent display of nature's exuberance!",
    '-6':   "The ocean feels wonderfully alive today. Waves roll in with a cheerful rhythm, and the sunlight plays across the water in beautiful patterns. There's something deeply refreshing about watching the sea move with such vibrant energy.",
    '-4':   "The ocean is a pleasant sight today. Its steady waves create a soothing rhythm, and the light reflecting off the surface adds a nice brightness to the scene. It's a good day to appreciate the water.",
    '-2':   "The ocean stretches out before me, calm and inviting. The waves move in their usual pattern, carrying foam toward shore. The light is decent today, reflecting off the surface in familiar ways.",
    '0':    "The ocean stretches out before me, vast and timeless. Its waves carry stories from distant shores, each crest a brief moment of light before settling back into the deep.",
    '2':    "The ocean lies before me, heavy and grey. Its waves move with a tired persistence, dragging themselves toward a shore that offers no comfort. There is a certain weariness in how it moves today.",
    '4':    "The ocean feels oppressively vast today. Its dark waters seem to swallow the light, and the waves crash with a hollow, echoing emptiness. Each recession leaves behind cold, forgotten things on the sand.",
    '6':    "The ocean is a void that stretches to the horizon, dark and indifferent. Its waves are not rhythmic but relentless, grinding against the shore with exhausting monotony. There is no comfort in its depths.",
    '8':    "The ocean is an abyss of sorrow, cold and merciless. Its waves crash with the weight of accumulated grief, each one a reminder of everything lost to its depths. The horizon offers no hope — only more of the same endless, desolate grey.",
  };

  function update(){
    const v = parseFloat(slider.value);
    const key = String(Math.round(v/2)*2); // snap to nearest even
    const closest = texts[key] || texts['0'];
    output.textContent = closest;

    const absV = Math.abs(v);
    if(v<-1){
      label.textContent = `α = ${v.toFixed(1)}σ (toward happy)`;
      label.style.color='var(--blue)';
      output.style.borderColor='rgba(96,165,250,.4)';
    } else if(v>1){
      label.textContent = `α = ${v>0?'+':''}${v.toFixed(1)}σ (toward sad)`;
      label.style.color='var(--red)';
      output.style.borderColor='rgba(248,113,113,.4)';
    } else {
      label.textContent = `α = 0.0σ (neutral)`;
      label.style.color='var(--accent2)';
      output.style.borderColor='var(--border)';
    }
  }

  slider.addEventListener('input',update);
  update();
})();

/* ═══════════════════════════════════════════
   MULTI-PROBE — Animated Radar Chart
   ═══════════════════════════════════════════ */
(function(){
  const c = document.getElementById('radar-canvas');
  if(!c) return;
  const ctx = c.getContext('2d');
  const W=c.width, H=c.height;
  const cx=W/2, cy=H/2, R=130;

  const probes = [
    {name:'empathy',color:'#f87171',target:.82},
    {name:'formal',color:'#60a5fa',target:.55},
    {name:'truthful',color:'#34d399',target:.91},
    {name:'confident',color:'#fbbf24',target:.68},
  ];
  const n = probes.length;
  const angleStep = Math.PI*2/n;

  // Animated values
  const vals = probes.map(()=>0);
  let running = false;

  const obs = new IntersectionObserver((entries)=>{
    entries.forEach(e=>{
      if(e.isIntersecting && !running){ running=true; animate(); }
      if(!e.isIntersecting) running=false;
    });
  },{threshold:.3});
  obs.observe(c);

  function animate(){
    if(!running) return;

    // Ease values toward targets
    probes.forEach((p,i)=>{vals[i] = lerp(vals[i], p.target, .03);});

    ctx.clearRect(0,0,W,H);

    // Grid circles
    for(let r=.25;r<=1;r+=.25){
      ctx.beginPath();
      for(let i=0;i<=n;i++){
        const a = -Math.PI/2 + i*angleStep;
        const x = cx + Math.cos(a)*R*r;
        const y = cy + Math.sin(a)*R*r;
        i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
      }
      ctx.closePath();
      ctx.strokeStyle='rgba(148,163,184,.1)';
      ctx.lineWidth=1;
      ctx.stroke();
    }

    // Axis lines + labels
    probes.forEach((p,i)=>{
      const a = -Math.PI/2 + i*angleStep;
      const x = cx + Math.cos(a)*R;
      const y = cy + Math.sin(a)*R;
      ctx.beginPath(); ctx.moveTo(cx,cy); ctx.lineTo(x,y);
      ctx.strokeStyle='rgba(148,163,184,.15)'; ctx.lineWidth=1; ctx.stroke();

      const lx = cx + Math.cos(a)*(R+22);
      const ly = cy + Math.sin(a)*(R+22);
      ctx.fillStyle=p.color;
      ctx.font='bold 10px Inter,sans-serif';
      ctx.textAlign='center';
      ctx.textBaseline='middle';
      ctx.fillText(p.name, lx, ly);
    });

    // Filled shape
    ctx.beginPath();
    probes.forEach((p,i)=>{
      const a = -Math.PI/2 + i*angleStep;
      const x = cx + Math.cos(a)*R*vals[i];
      const y = cy + Math.sin(a)*R*vals[i];
      i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
    });
    ctx.closePath();
    ctx.fillStyle='rgba(99,102,241,.15)';
    ctx.fill();
    ctx.strokeStyle='rgba(99,102,241,.6)';
    ctx.lineWidth=2;
    ctx.stroke();

    // Dots
    probes.forEach((p,i)=>{
      const a = -Math.PI/2 + i*angleStep;
      const x = cx + Math.cos(a)*R*vals[i];
      const y = cy + Math.sin(a)*R*vals[i];
      ctx.beginPath(); ctx.arc(x,y,5,0,Math.PI*2);
      ctx.fillStyle=p.color;
      ctx.shadowColor=p.color;
      ctx.shadowBlur=10;
      ctx.fill();
      ctx.shadowBlur=0;

      // Value label
      ctx.fillStyle='rgba(226,232,240,.7)';
      ctx.font='9px JetBrains Mono,monospace';
      ctx.textAlign='center';
      ctx.fillText(vals[i].toFixed(2), x, y - 12);
    });

    requestAnimationFrame(animate);
  }
})();

/* ═══════════════════════════════════════════
   EVAL PLOTS — Mini animated charts
   ═══════════════════════════════════════════ */
(function(){
  // Accuracy vs Alpha
  const c1 = document.getElementById('eval-acc-canvas');
  if(c1) drawMiniLineChart(c1, [-8,-4,0,4,8], [.92,.88,.85,.72,.55], '#6366f1','Accuracy');

  // Score vs Alpha
  const c2 = document.getElementById('eval-score-canvas');
  if(c2) drawMiniLineChart(c2, [-8,-4,0,4,8], [-.35,-.12,.05,.28,.62], '#34d399','Mean Score');

  // Score by correctness (bar chart)
  const c3 = document.getElementById('eval-correct-canvas');
  if(c3) drawMiniBarChart(c3, ['Correct','Wrong'], [.12,.45], ['#34d399','#f87171']);

  function drawMiniLineChart(canvas, xs, ys, color, title){
    const ctx = canvas.getContext('2d');
    const W=canvas.width, H=canvas.height;
    const pad={t:28,b:24,l:30,r:10};
    const gW=W-pad.l-pad.r, gH=H-pad.t-pad.b;

    const yMin = Math.min(...ys)-.1;
    const yMax = Math.max(...ys)+.1;
    const xMin = Math.min(...xs);
    const xMax = Math.max(...xs);

    let progress = 0;
    let running = false;

    const obs = new IntersectionObserver((entries)=>{
      entries.forEach(e=>{
        if(e.isIntersecting && !running){ running=true; animate(); }
        if(!e.isIntersecting) running=false;
      });
    },{threshold:.5});
    obs.observe(canvas);

    function animate(){
      if(!running) return;
      progress = Math.min(1, progress+.025);
      ctx.clearRect(0,0,W,H);

      // Title
      ctx.fillStyle='rgba(226,232,240,.7)';
      ctx.font='bold 10px Inter,sans-serif';
      ctx.textAlign='left';
      ctx.fillText(title, pad.l, 14);

      // Axes
      ctx.strokeStyle='rgba(148,163,184,.2)';
      ctx.lineWidth=1;
      ctx.beginPath();
      ctx.moveTo(pad.l,pad.t);ctx.lineTo(pad.l,H-pad.b);ctx.lineTo(W-pad.r,H-pad.b);
      ctx.stroke();

      // Y labels
      ctx.fillStyle='rgba(148,163,184,.4)';
      ctx.font='8px JetBrains Mono,monospace';
      ctx.textAlign='right';
      for(let i=0;i<=4;i++){
        const v = yMin + (yMax-yMin)*i/4;
        const y = H-pad.b-(i/4)*gH;
        ctx.fillText(v.toFixed(1), pad.l-4, y+3);
      }

      // X labels
      ctx.textAlign='center';
      xs.forEach((x,i)=>{
        const px = pad.l + ((x-xMin)/(xMax-xMin))*gW;
        ctx.fillText(x, px, H-pad.b+14);
      });

      // Line
      const n = Math.ceil(xs.length * progress);
      ctx.beginPath();
      for(let i=0;i<n;i++){
        const px = pad.l + ((xs[i]-xMin)/(xMax-xMin))*gW;
        const py = H-pad.b - ((ys[i]-yMin)/(yMax-yMin))*gH;
        i===0?ctx.moveTo(px,py):ctx.lineTo(px,py);
      }
      ctx.strokeStyle=color;
      ctx.lineWidth=2;
      ctx.stroke();

      // Dots
      for(let i=0;i<n;i++){
        const px = pad.l + ((xs[i]-xMin)/(xMax-xMin))*gW;
        const py = H-pad.b - ((ys[i]-yMin)/(yMax-yMin))*gH;
        ctx.beginPath();ctx.arc(px,py,3,0,Math.PI*2);
        ctx.fillStyle=color;ctx.fill();
      }

      if(progress<1) requestAnimationFrame(animate);
    }
  }

  function drawMiniBarChart(canvas, labels, values, colors){
    const ctx = canvas.getContext('2d');
    const W=canvas.width, H=canvas.height;
    const pad={t:28,b:24,l:30,r:10};
    const gH=H-pad.t-pad.b;

    let progress = 0;
    let running = false;

    const obs = new IntersectionObserver((entries)=>{
      entries.forEach(e=>{
        if(e.isIntersecting && !running){ running=true; animate(); }
        if(!e.isIntersecting) running=false;
      });
    },{threshold:.5});
    obs.observe(canvas);

    function animate(){
      if(!running) return;
      progress = Math.min(1, progress+.03);
      ctx.clearRect(0,0,W,H);

      ctx.fillStyle='rgba(226,232,240,.7)';
      ctx.font='bold 10px Inter,sans-serif';
      ctx.textAlign='left';
      ctx.fillText('Score by Correctness', pad.l, 14);

      const maxV = Math.max(...values)+.1;
      const barW = 50;
      const gap = 30;
      const totalW = labels.length*barW + (labels.length-1)*gap;
      const startX = pad.l + (W-pad.l-pad.r-totalW)/2;

      labels.forEach((l,i)=>{
        const x = startX + i*(barW+gap);
        const h = (values[i]/maxV)*gH*progress;
        const y = H-pad.b-h;

        ctx.fillStyle=colors[i];
        ctx.globalAlpha=.7;
        safeRoundRect(ctx,x,y,barW,h,[4,4,0,0]);
        ctx.fill();
        ctx.globalAlpha=1;

        ctx.fillStyle='rgba(148,163,184,.6)';
        ctx.font='9px Inter,sans-serif';
        ctx.textAlign='center';
        ctx.fillText(l, x+barW/2, H-pad.b+14);

        if(progress>.5){
          ctx.fillStyle=colors[i];
          ctx.font='bold 10px JetBrains Mono,monospace';
          ctx.fillText((values[i]*progress).toFixed(2), x+barW/2, y-6);
        }
      });

      if(progress<1) requestAnimationFrame(animate);
    }
  }
})();

/* ═══════════════════════════════════════════
   Smooth scroll for nav links
   ═══════════════════════════════════════════ */
$$('a[href^="#"]').forEach(a=>{
  a.addEventListener('click',e=>{
    const target = document.querySelector(a.getAttribute('href'));
    if(target){
      e.preventDefault();
      target.scrollIntoView({behavior:'smooth'});
      $('#nav-links').classList.remove('open');
    }
  });
});
