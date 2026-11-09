(() => {
  const canvas = document.getElementById('scene-canvas');
  const ctx = canvas.getContext('2d');
  const modeSelect = document.getElementById('mode-select');
  const playButton = document.getElementById('play-button');
  const stepper = document.getElementById('stepper');
  const viewStatus = document.getElementById('view-status');
  const pointStatus = document.getElementById('point-status');
  const mapStatus = document.getElementById('map-status');
  const stepNumber = document.getElementById('step-number');
  const stepTitle = document.getElementById('step-title');
  const stepDescription = document.getElementById('step-description');
  const algorithmList = document.getElementById('algorithm-list');
  const dataLabel = document.getElementById('data-label');
  const dataFrame = document.getElementById('data-frame');
  const dataPreview = document.getElementById('data-preview');
  const stageTakeaway = document.getElementById('stage-takeaway');

  const stages = [
    {
      short: 'Initial RGB-D', title: 'Initial overhead observation',
      description: 'Gemini 335 captures aligned RGB and depth. Every valid depth pixel is back-projected into the robot-base coordinate frame.',
      algorithms: ['RGB–depth alignment', 'Back-projection', 'TF2'],
      takeaway: 'A single depth frame creates a measurable but incomplete 2.5D surface map.',
      data: { camera: 'Gemini 335', frame: 'robot_base', depth: '1280×800', map_state: 'partial' }
    },
    {
      short: 'Left view', title: 'Acquire a left-front wrist view',
      description: 'The SO-101 moves the wrist-mounted Gemini 305 to a safe, reachable pose and measures surfaces hidden from above.',
      algorithms: ['Forward kinematics', 'Hand–eye calibration', 'Safe waypoint'],
      takeaway: 'Changing viewpoint reveals side surfaces instead of asking a model to hallucinate them.',
      data: { camera: 'Gemini 305', pose_source: 'FK × hand_eye', view: 'left_front', action: 'capture_rgbd' }
    },
    {
      short: 'Right view', title: 'Acquire an opposite wrist view',
      description: 'A second wrist observation covers the opposite faces and provides overlapping geometry for registration checks.',
      algorithms: ['Reachability check', 'RGB-D capture', 'Overlap test'],
      takeaway: 'Opposing views reduce one-sided centroid and dimension bias.',
      data: { camera: 'Gemini 305', view: 'right_front', overlap_required: true, action: 'capture_rgbd' }
    },
    {
      short: 'Fuse map', title: 'Register and fuse partial clouds',
      description: 'Robot transforms provide coarse alignment. Constrained ICP corrects small residual errors before voxel or TSDF fusion.',
      algorithms: ['TF coarse alignment', 'Constrained ICP', 'Voxel / TSDF'],
      takeaway: 'The result is a task-specific local 3D map, not a permanent or fully observed world model.',
      data: { inputs: ['overhead', 'left_front', 'right_front', 'side_check'], registration: 'TF + constrained ICP', fusion: 'TSDF', unknown_space: 'preserved' }
    },
    {
      short: 'Scene graph', title: 'Build an object-level 3D scene graph',
      description: 'The table plane is removed, object clusters are fitted with cuboids, and geometric relations are computed in one coordinate frame.',
      algorithms: ['RANSAC plane', 'DBSCAN / clustering', 'Cuboid fit', 'Geometric relations'],
      takeaway: 'The map becomes an inspectable set of objects and verifiable spatial relations.',
      data: { objects: 3, relations: [['blue_cube', 'left_of', 'red_cube'], ['green_block', 'behind', 'red_cube']], confidence_source: 'geometry' }
    },
    {
      short: 'Ground command', title: 'Ground the spatial language command',
      description: 'Language is converted into a symbolic graph query. The target is selected using measured geometry rather than generated coordinates.',
      algorithms: ['Language → JSON', 'Graph query', 'Ambiguity check'],
      takeaway: '“The cube left of the red cube” resolves to blue_cube with an explicit geometric reason.',
      data: { command: 'Pass the cube left of the red cube', query: { type: 'cube', relation: 'left_of', reference: 'red_cube' }, selected: 'blue_cube' }
    },
    {
      short: 'Plan & verify', title: 'Plan, execute and update the map',
      description: 'A grasp pose and collision-aware trajectory are generated. After motion, a fresh RGB-D observation updates the changed region.',
      algorithms: ['Grasp pose', 'MoveIt 2', 'ROS 2 control', 'Post-action verification'],
      takeaway: 'The system closes the loop: perceive, reason, act, verify, then update instead of trusting a stale map.',
      data: { target: 'blue_cube', sequence: ['pre_grasp', 'grasp', 'handover', 'release'], verification: 'fresh_rgbd', map_update: 'changed_region' }
    }
  ];

  const state = { stage: 0, mode: 'active', yaw: -0.72, pitch: 0.72, zoom: 1, selected: null, drag: null, playing: false, phase: 0 };
  const objects = [
    { id: 'red_cube', label: 'red cube', center: [0.26, 0.16, 0.10], size: [0.22,0.22,0.20], color: '#ff6c7b' },
    { id: 'blue_cube', label: 'blue cube', center: [-0.25, 0.08, 0.085], size: [0.18,0.18,0.17], color: '#4f8dff' },
    { id: 'green_block', label: 'green block', center: [0.02,-0.30,0.075], size: [0.30,0.16,0.15], color: '#64e3a1' }
  ];
  const sensorViews = [
    { name: 'Gemini 335 · overhead', pos: [0,-0.05,1.75] },
    { name: 'Gemini 305 · left-front', pos: [-1.05,-1.05,0.83] },
    { name: 'Gemini 305 · right-front', pos: [1.10,-0.92,0.78] },
    { name: 'Gemini 305 · side check', pos: [1.22,0.56,0.66] }
  ];
  const faces = [
    { axis:0, side:1, normal:[1,0,0] }, { axis:0, side:-1, normal:[-1,0,0] },
    { axis:1, side:1, normal:[0,1,0] }, { axis:1, side:-1, normal:[0,-1,0] },
    { axis:2, side:1, normal:[0,0,1] }, { axis:2, side:-1, normal:[0,0,-1] }
  ];
  const edges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
  const dot=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
  const sub=(a,b)=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]];
  const add=(a,b)=>[a[0]+b[0],a[1]+b[1],a[2]+b[2]];
  const mul=(a,s)=>[a[0]*s,a[1]*s,a[2]*s];
  const len=a=>Math.hypot(a[0],a[1],a[2]);
  const norm=a=>mul(a,1/(len(a)||1));
  const lerp=(a,b,t)=>a.map((v,i)=>v+(b[i]-v)*t);

  function makePoints(object) {
    const out=[], n=9;
    faces.forEach((face,faceIndex)=>{
      for(let i=0;i<n;i++) for(let j=0;j<n;j++) {
        const q=[-0.5,-0.5,-0.5]; q[face.axis]=face.side*0.5;
        const other=[0,1,2].filter(k=>k!==face.axis);
        q[other[0]]=-0.5+(i+0.5)/n; q[other[1]]=-0.5+(j+0.5)/n;
        out.push({ p:object.center.map((v,k)=>v+q[k]*object.size[k]), normal:face.normal, key:`${object.id}-${faceIndex}-${i}-${j}`, object });
      }
    });
    return out;
  }
  const allPoints=objects.flatMap(makePoints);
  const visibleFrom=(point,view)=>dot(point.normal,norm(sub(view.pos,point.p)))>0.12;
  function viewCount() {
    if(state.mode==='single') return 1;
    if(state.stage===0) return 1;
    if(state.stage===1) return 2;
    if(state.stage===2) return 3;
    return 4;
  }
  function measuredPoints() {
    const views=sensorViews.slice(0,viewCount()), seen=new Map();
    allPoints.forEach(p=>{if(views.some(v=>visibleFrom(p,v)))seen.set(p.key,p);});
    return [...seen.values()];
  }
  const coverage=()=>Math.round(measuredPoints().length/allPoints.length*100);

  function buildStepper() {
    stepper.innerHTML='';
    stages.forEach((step,index)=>{
      const button=document.createElement('button'); button.type='button'; button.className='step-button';
      button.innerHTML=`<span>${String(index+1).padStart(2,'0')}</span><strong>${step.short}</strong>`;
      button.addEventListener('click',()=>{stop();setStage(index);}); stepper.appendChild(button);
    });
  }
  function setStage(index) {
    state.stage=Math.max(0,Math.min(stages.length-1,index));
    [...stepper.children].forEach((button,i)=>button.classList.toggle('active',i===state.stage));
    updatePanel(); draw();
  }
  function updatePanel() {
    const step=stages[state.stage];
    stepNumber.textContent=`STEP ${String(state.stage+1).padStart(2,'0')} / ${String(stages.length).padStart(2,'0')}`;
    stepTitle.textContent=step.title;
    stepDescription.textContent=(state.mode==='single'&&state.stage>0&&state.stage<4?'Single-view comparison keeps only the overhead measurement. ':'')+step.description;
    algorithmList.innerHTML=step.algorithms.map(item=>`<span>${item}</span>`).join('');
    dataLabel.textContent=state.stage===5?'Structured query':'Pipeline state'; dataFrame.textContent=state.stage<4?`view ${viewCount()}`:'robot_base';
    dataPreview.textContent=JSON.stringify(step.data,null,2); stageTakeaway.textContent=step.takeaway;
    const currentView=state.mode==='single'?sensorViews[0]:sensorViews[Math.min(state.stage,3)];
    viewStatus.textContent=`${viewCount()} view${viewCount()===1?'':'s'} · ${currentView.name}`;
    pointStatus.textContent=`${measuredPoints().length.toLocaleString()} simulated samples`;
    mapStatus.textContent=state.stage<3?`${coverage()}% visible surface`:state.stage===3?'Local 3D map':state.stage===4?'3D scene graph':state.stage===5?'Target: blue cube':'Map update after action';
  }

  function resize() {
    const rect=canvas.getBoundingClientRect(), dpr=Math.min(devicePixelRatio||1,2);
    canvas.width=Math.round(rect.width*dpr); canvas.height=Math.round(rect.height*dpr); ctx.setTransform(dpr,0,0,dpr,0,0); draw();
  }
  function project(p) {
    const rect=canvas.getBoundingClientRect(),cy=Math.cos(state.yaw),sy=Math.sin(state.yaw),cp=Math.cos(state.pitch),sp=Math.sin(state.pitch);
    const x1=cy*p[0]-sy*p[1],y1=sy*p[0]+cy*p[1],x2=x1,y2=cp*p[2]-sp*y1,z2=sp*p[2]+cp*y1;
    const scale=Math.min(rect.width,rect.height)*0.41*state.zoom;
    return {x:rect.width*.5+x2*scale,y:rect.height*.62-y2*scale,depth:z2};
  }
  function line3(a,b,color,width=1,dash=[]) {
    const pa=project(a),pb=project(b);ctx.save();ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash(dash);ctx.beginPath();ctx.moveTo(pa.x,pa.y);ctx.lineTo(pb.x,pb.y);ctx.stroke();ctx.restore();
  }
  function label3(p,text,color,offset=[0,0]) { const q=project(p);ctx.fillStyle=color;ctx.font='12px ui-monospace, SFMono-Regular, Menlo, monospace';ctx.fillText(text,q.x+offset[0],q.y+offset[1]); }
  function corners(o) {
    const [sx,sy,sz]=o.size.map(v=>v/2),[cx,cy,cz]=o.center;
    return [[-sx,-sy,-sz],[sx,-sy,-sz],[sx,sy,-sz],[-sx,sy,-sz],[-sx,-sy,sz],[sx,-sy,sz],[sx,sy,sz],[-sx,sy,sz]].map(v=>[cx+v[0],cy+v[1],cz+v[2]]);
  }
  function cube(o,color,width=1,dash=[]) { const c=corners(o);edges.forEach(e=>line3(c[e[0]],c[e[1]],color,width,dash)); }
  function arrow3(a,b,color,label) {
    const pa=project(a),pb=project(b),angle=Math.atan2(pb.y-pa.y,pb.x-pa.x);ctx.save();ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(pa.x,pa.y);ctx.lineTo(pb.x,pb.y);ctx.stroke();ctx.beginPath();ctx.moveTo(pb.x,pb.y);ctx.lineTo(pb.x-9*Math.cos(angle-.45),pb.y-9*Math.sin(angle-.45));ctx.lineTo(pb.x-9*Math.cos(angle+.45),pb.y-9*Math.sin(angle+.45));ctx.closePath();ctx.fill();ctx.font='12px ui-monospace,monospace';ctx.fillText(label,(pa.x+pb.x)/2+5,(pa.y+pb.y)/2-5);ctx.restore();
  }
  function drawTable() {
    const min=-.72,max=.72,step=.18;for(let v=min;v<=max+.001;v+=step){line3([v,min,0],[v,max,0],'#223847',.7);line3([min,v,0],[max,v,0],'#223847',.7);}
    const ring=[[min,min],[max,min],[max,max],[min,max],[min,min]];for(let i=0;i<ring.length-1;i++)line3([ring[i][0],ring[i][1],0],[ring[i+1][0],ring[i+1][1],0],'#688297',1);
  }
  function drawPoints() {
    measuredPoints().map(pt=>({pt,q:project(pt.p)})).sort((a,b)=>a.q.depth-b.q.depth).forEach(({pt,q})=>{ctx.globalAlpha=state.stage<4?.92:.44;ctx.fillStyle=pt.object.color;ctx.beginPath();ctx.arc(q.x,q.y,state.stage<3?1.7:1.35,0,Math.PI*2);ctx.fill();});ctx.globalAlpha=1;
  }
  function currentSensor(){return state.mode==='single'?sensorViews[0]:sensorViews[Math.min(state.stage,3)];}
  function drawSensor() {
    if(state.stage>3)return;const sensor=currentSensor(),p=project(sensor.pos);line3(sensor.pos,[0,0,.05],'#ffac4d',1.5,[5,4]);ctx.save();ctx.translate(p.x,p.y);ctx.rotate(-.18);ctx.fillStyle='#ffac4d';ctx.strokeStyle='#ecf5fb';ctx.lineWidth=1;ctx.beginPath();ctx.roundRect(-15,-9,30,18,4);ctx.fill();ctx.stroke();ctx.beginPath();ctx.arc(0,0,4,0,Math.PI*2);ctx.fillStyle='#071018';ctx.fill();ctx.restore();label3(sensor.pos,sensor.name,'#ecf5fb',[18,-10]);
  }
  function drawGraph() {
    if(state.stage<4)return;objects.forEach(o=>{const target=state.stage>=5&&o.id==='blue_cube';cube(o,target?'#64e3a1':'#d7e8f2',target?3:1.5,target?[]:[4,3]);label3(add(o.center,[0,0,o.size[2]/2+.06]),o.id,target?'#64e3a1':'#ecf5fb',[-28,0]);});arrow3(add(objects[1].center,[0,0,.17]),add(objects[0].center,[0,0,.17]),'#b58cff','left_of');arrow3(add(objects[2].center,[0,0,.20]),add(objects[0].center,[0,0,.20]),'#b58cff','behind');
  }
  function drawRobot() {
    if(state.stage<6)return;const joints=[[-.68,.55,0],[-.55,.45,.28],[-.33,.32,.48],[-.12,.20,.39]];for(let i=0;i<joints.length-1;i++)line3(joints[i],joints[i+1],'#d7e8f2',5);joints.forEach(j=>{const p=project(j);ctx.fillStyle='#071018';ctx.strokeStyle='#d7e8f2';ctx.lineWidth=2;ctx.beginPath();ctx.arc(p.x,p.y,5,0,Math.PI*2);ctx.fill();ctx.stroke();});const path=[joints.at(-1),[-.25,.08,.38],[-.25,.08,.22],[-.04,-.56,.34]];for(let i=0;i<path.length-1;i++)line3(path[i],path[i+1],'#b58cff',2,[7,5]);const t=(Math.sin(state.phase)+1)/2*(path.length-1),idx=Math.min(path.length-2,Math.floor(t)),hand=lerp(path[idx],path[idx+1],t-idx),p=project(hand);ctx.fillStyle='#64e3a1';ctx.beginPath();ctx.arc(p.x,p.y,6,0,Math.PI*2);ctx.fill();
  }
  function draw() {
    const r=canvas.getBoundingClientRect();if(!r.width)return;ctx.clearRect(0,0,r.width,r.height);const g=ctx.createRadialGradient(r.width*.48,r.height*.48,20,r.width*.48,r.height*.48,r.width*.7);g.addColorStop(0,'#102331');g.addColorStop(1,'#08131c');ctx.fillStyle=g;ctx.fillRect(0,0,r.width,r.height);drawTable();drawPoints();drawSensor();drawGraph();drawRobot();if(state.stage<4)objects.forEach(o=>cube(o,'#294454',.8,[3,4]));if(state.selected){const o=objects.find(x=>x.id===state.selected);if(o)cube(o,'#ffdc69',3);}
  }
  function hitObject(x,y) {let best=null,distance=36;objects.forEach(o=>{const p=project(o.center),d=Math.hypot(x-p.x,y-p.y);if(d<distance){best=o;distance=d;}});return best;}

  let timer=null;
  function stop(){state.playing=false;playButton.textContent='▶ Play route';if(timer){clearInterval(timer);timer=null;}}
  function play(){state.playing=true;playButton.textContent='Ⅱ Pause';if(state.stage===stages.length-1)setStage(0);timer=setInterval(()=>{if(state.stage===stages.length-1){stop();return;}setStage(state.stage+1);},2100);}
  playButton.addEventListener('click',()=>state.playing?stop():play());
  document.getElementById('previous-button').addEventListener('click',()=>{stop();setStage(state.stage-1);});
  document.getElementById('next-button').addEventListener('click',()=>{stop();setStage(state.stage+1);});
  document.getElementById('reset-view').addEventListener('click',()=>{state.yaw=-.72;state.pitch=.72;state.zoom=1;draw();});
  modeSelect.addEventListener('change',()=>{state.mode=modeSelect.value;stop();updatePanel();draw();});
  canvas.addEventListener('pointerdown',e=>{canvas.setPointerCapture(e.pointerId);state.drag={x:e.clientX,y:e.clientY,yaw:state.yaw,pitch:state.pitch,moved:false};});
  canvas.addEventListener('pointermove',e=>{if(!state.drag)return;const dx=e.clientX-state.drag.x,dy=e.clientY-state.drag.y;if(Math.abs(dx)+Math.abs(dy)>3)state.drag.moved=true;state.yaw=state.drag.yaw+dx*.008;state.pitch=Math.max(.18,Math.min(1.35,state.drag.pitch+dy*.006));draw();});
  canvas.addEventListener('pointerup',e=>{if(state.drag&&!state.drag.moved){const r=canvas.getBoundingClientRect(),o=hitObject(e.clientX-r.left,e.clientY-r.top);state.selected=o?o.id:null;if(o){dataLabel.textContent=`Selected · ${o.label}`;dataPreview.textContent=JSON.stringify({id:o.id,center_base_m:o.center,dimensions_m:o.size,observation:state.stage>=4?'fused_and_fitted':'partial_surface'},null,2);}else updatePanel();draw();}state.drag=null;});
  canvas.addEventListener('wheel',e=>{e.preventDefault();state.zoom=Math.max(.68,Math.min(1.6,state.zoom*Math.exp(-e.deltaY*.001)));draw();},{passive:false});
  new ResizeObserver(resize).observe(canvas);
  function animate(){state.phase+=.025;if(state.stage===6)draw();requestAnimationFrame(animate);}
  buildStepper();setStage(0);resize();animate();
})();
