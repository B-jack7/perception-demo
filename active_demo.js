(() => {
  const canvas = document.getElementById('scene-canvas');
  const ctx = canvas.getContext('2d');
  const modeSelect = document.getElementById('mode-select');
  const playButton = document.getElementById('play-button');
  const stepper = document.getElementById('stepper');
  const ui = Object.fromEntries([
    'view-status','point-status','map-status','step-number','step-title','step-description',
    'algorithm-list','data-label','data-frame','data-preview','stage-takeaway'
  ].map(id => [id, document.getElementById(id)]));

  const stages = [
    {
      short:'Initial RGB-D', title:'Observe the tabletop from above',
      description:'Gemini 335 measures the visible surfaces of a cup, cube and bottle. Depth pixels are back-projected into the robot-base frame.',
      algorithms:['RGB–depth alignment','Back-projection','TF2'],
      takeaway:'The first frame provides metric surface coordinates, but the backs and sides remain unknown.',
      data:{camera:'Gemini 335',objects_visible:3,frame:'robot_base',map_state:'partial_2.5D'}
    },
    {
      short:'Left scan', title:'Reveal hidden surfaces from the left',
      description:'The wrist-mounted Gemini 305 moves to a safe left-front pose. The cup handle and side walls become more observable.',
      algorithms:['Forward kinematics','Hand–eye transform','Safe waypoint'],
      takeaway:'A physical viewpoint change adds measured geometry instead of guessed geometry.',
      data:{camera:'Gemini 305',view:'left_front',focus:['cup_handle','bottle_side'],action:'capture_rgbd'}
    },
    {
      short:'Right scan', title:'Acquire the opposite wrist view',
      description:'A right-front observation adds the opposing faces and overlapping regions needed to verify registration.',
      algorithms:['Reachability check','RGB-D capture','Overlap test'],
      takeaway:'Opposing views reduce one-sided centroid and dimension bias.',
      data:{camera:'Gemini 305',view:'right_front',overlap_required:true,action:'capture_rgbd'}
    },
    {
      short:'Fuse map', title:'Register and fuse the partial point clouds',
      description:'Robot transforms provide coarse alignment. Constrained ICP corrects small residuals before voxel or TSDF fusion.',
      algorithms:['TF coarse alignment','Constrained ICP','Voxel / TSDF'],
      takeaway:'The fused result is a task-specific local 3D map. Unseen space remains unknown.',
      data:{views:['overhead','left_front','right_front','side_check'],registration:'TF + constrained ICP',fusion:'TSDF',unknown_space:'preserved'}
    },
    {
      short:'Instances', title:'Separate the table into object instances',
      description:'The table plane is removed. Connected 3D regions are clustered into three independent objects before any semantic name is assigned.',
      algorithms:['RANSAC plane removal','DBSCAN','Region growing','3D bounding volumes'],
      takeaway:'Geometry answers “where is each object?” before the system decides “what is it?”.',
      data:{instances:[{id:'object_1',points:708},{id:'object_2',points:486},{id:'object_3',points:912}],frame:'robot_base'}
    },
    {
      short:'Recognize', title:'Classify cup, cube and bottle',
      description:'Each 3D cluster is projected into the RGB views. Shape evidence, colour and multi-view CLIP/SigLIP scores are fused into a semantic label.',
      algorithms:['3D primitive fitting','Multi-view RGB crops','CLIP / SigLIP','Evidence fusion'],
      takeaway:'Point clouds provide shape and position; RGB appearance provides the semantic category.',
      data:{
        object_1:{label:'cup',geometry:'hollow cylinder + handle',semantic_score:0.91},
        object_2:{label:'cube',geometry:'six planes',semantic_score:0.96},
        object_3:{label:'bottle',geometry:'body + shoulder + neck',semantic_score:0.89}
      }
    },
    {
      short:'Ground command', title:'Resolve the spatial language command',
      description:'Objects become nodes in a metric scene graph. A language instruction is converted into a graph query and selects the measured cup.',
      algorithms:['3D scene graph','Language → JSON','Graph query','Ambiguity check'],
      takeaway:'“Move the cup on the left to the right side” resolves to cup_1 with an inspectable geometric reason.',
      data:{command:'Move the cup on the left to the right side',query:{category:'cup',selector:'leftmost',destination:'right_region',reference_frame:'robot_base'},selected:'cup_1'}
    },
    {
      short:'Move & update', title:'Plan the motion and update the map',
      description:'A grasp pose and collision-aware trajectory move the selected cup. A fresh RGB-D observation replaces the changed region of the map.',
      algorithms:['Grasp pose','MoveIt 2','ROS 2 control','Post-action verification'],
      takeaway:'The map is not trusted forever: perceive, act, verify and update closes the loop.',
      data:{target:'cup_1',sequence:['pre_grasp','grasp','move_to_right_region','release'],verification:'fresh_rgbd',map_update:'changed_region'}
    }
  ];

  const state={stage:0,mode:'active',yaw:-0.72,pitch:0.72,zoom:1,selected:null,drag:null,playing:false,phase:0};
  const objects=[
    {id:'cup_1',instance:'object_1',label:'cup',kind:'cup',center:[-0.30,0.07,0.10],size:[0.26,0.20,0.20],color:'#ef6e7e',score:0.91,evidence:'hollow cylinder + handle'},
    {id:'cube_1',instance:'object_2',label:'cube',kind:'cube',center:[0.15,0.17,0.09],size:[0.18,0.18,0.18],color:'#4f8dff',score:0.96,evidence:'six orthogonal planes'},
    {id:'bottle_1',instance:'object_3',label:'bottle',kind:'bottle',center:[0.20,-0.25,0.15],size:[0.16,0.16,0.30],color:'#64e3a1',score:0.89,evidence:'cylindrical body + neck'}
  ];
  const sensorViews=[
    {name:'Gemini 335 · overhead',pos:[0,-0.05,1.75]},
    {name:'Gemini 305 · left-front',pos:[-1.05,-1.05,0.83]},
    {name:'Gemini 305 · right-front',pos:[1.10,-0.92,0.78]},
    {name:'Gemini 305 · side check',pos:[1.22,0.56,0.66]}
  ];
  const boxFaces=[
    {axis:0,side:1,normal:[1,0,0]},{axis:0,side:-1,normal:[-1,0,0]},
    {axis:1,side:1,normal:[0,1,0]},{axis:1,side:-1,normal:[0,-1,0]},
    {axis:2,side:1,normal:[0,0,1]},{axis:2,side:-1,normal:[0,0,-1]}
  ];
  const boxEdges=[[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
  const dot=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
  const sub=(a,b)=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]];
  const add=(a,b)=>[a[0]+b[0],a[1]+b[1],a[2]+b[2]];
  const mul=(a,s)=>[a[0]*s,a[1]*s,a[2]*s];
  const norm=a=>{const l=Math.hypot(...a)||1;return mul(a,1/l);};
  const lerp=(a,b,t)=>a.map((v,i)=>v+(b[i]-v)*t);

  function cubePoints(o) {
    const out=[],n=9;
    boxFaces.forEach((face,f)=>{for(let i=0;i<n;i++)for(let j=0;j<n;j++){
      const q=[-.5,-.5,-.5];q[face.axis]=face.side*.5;const other=[0,1,2].filter(k=>k!==face.axis);
      q[other[0]]=-.5+(i+.5)/n;q[other[1]]=-.5+(j+.5)/n;
      out.push({p:o.center.map((v,k)=>v+q[k]*o.size[k]),normal:face.normal,key:`${o.id}-box-${f}-${i}-${j}`,object:o});
    }});return out;
  }
  function cylinderShell(o,radius,height,zBase,prefix,steps=48,layers=10) {
    const out=[];for(let a=0;a<steps;a++){const t=2*Math.PI*a/steps,n=[Math.cos(t),Math.sin(t),0];for(let z=0;z<layers;z++){
      out.push({p:[o.center[0]+radius*Math.cos(t),o.center[1]+radius*Math.sin(t),zBase+height*(z+.5)/layers],normal:n,key:`${o.id}-${prefix}-${a}-${z}`,object:o});
    }}return out;
  }
  function diskRing(o,radius,z,prefix,steps=48,rings=3) {
    const out=[];for(let a=0;a<steps;a++){const t=2*Math.PI*a/steps;for(let k=0;k<rings;k++){
      const r=radius*(.68+.30*(k+1)/rings);out.push({p:[o.center[0]+r*Math.cos(t),o.center[1]+r*Math.sin(t),z],normal:[0,0,1],key:`${o.id}-${prefix}-${a}-${k}`,object:o});
    }}return out;
  }
  function cupPoints(o) {
    const out=[...cylinderShell(o,.095,.18,.01,'cup-side',48,10),...diskRing(o,.095,.19,'cup-rim',48,2)];
    const hx=o.center[0]-.125,hz=.105;for(let a=0;a<42;a++){const t=2*Math.PI*a/42;for(let k=-1;k<=1;k++){
      const p=[hx+.062*Math.cos(t),o.center[1]+k*.008,hz+.065*Math.sin(t)];const normal=norm([Math.cos(t),k*.3,Math.sin(t)]);
      out.push({p,normal,key:`${o.id}-handle-${a}-${k}`,object:o});
    }}return out;
  }
  function bottlePoints(o) {
    const out=[...cylinderShell(o,.078,.19,.005,'body',48,10),...cylinderShell(o,.034,.07,.225,'neck',48,6),...diskRing(o,.034,.298,'cap',48,3)];
    for(let a=0;a<48;a++){const t=2*Math.PI*a/48;for(let k=0;k<5;k++){
      const u=(k+.5)/5,r=.078+(.034-.078)*u,z=.19+.035*u;
      out.push({p:[o.center[0]+r*Math.cos(t),o.center[1]+r*Math.sin(t),z],normal:norm([Math.cos(t),Math.sin(t),.55]),key:`${o.id}-shoulder-${a}-${k}`,object:o});
    }}return out;
  }
  const allPoints=objects.flatMap(o=>o.kind==='cube'?cubePoints(o):o.kind==='cup'?cupPoints(o):bottlePoints(o));
  const visibleFrom=(point,view)=>dot(point.normal,norm(sub(view.pos,point.p)))>.08;
  function viewCount(){if(state.mode==='single')return 1;if(state.stage===0)return 1;if(state.stage===1)return 2;if(state.stage===2)return 3;return 4;}
  function moveProgress(){return state.stage===7?(Math.sin(state.phase)+1)/2:0;}
  function cupOffset(){return mul([.72,-.04,0],moveProgress());}
  function measuredPoints(){const views=sensorViews.slice(0,viewCount()),seen=new Map();allPoints.forEach(p=>{if(views.some(v=>visibleFrom(p,v)))seen.set(p.key,p);});return [...seen.values()];}
  const coverage=()=>Math.round(measuredPoints().length/allPoints.length*100);

  function buildStepper(){stepper.innerHTML='';stages.forEach((step,index)=>{const b=document.createElement('button');b.type='button';b.className='step-button';b.innerHTML=`<span>${String(index+1).padStart(2,'0')}</span><strong>${step.short}</strong>`;b.addEventListener('click',()=>{stop();setStage(index);});stepper.appendChild(b);});}
  function setStage(index){state.stage=Math.max(0,Math.min(stages.length-1,index));state.selected=null;[...stepper.children].forEach((b,i)=>b.classList.toggle('active',i===state.stage));updatePanel();draw();}
  function updatePanel(){const step=stages[state.stage];ui['step-number'].textContent=`STEP ${String(state.stage+1).padStart(2,'0')} / ${String(stages.length).padStart(2,'0')}`;ui['step-title'].textContent=step.title;ui['step-description'].textContent=(state.mode==='single'&&state.stage>0&&state.stage<4?'Single-view comparison keeps only the overhead measurement. ':'')+step.description;ui['algorithm-list'].innerHTML=step.algorithms.map(x=>`<span>${x}</span>`).join('');ui['data-label'].textContent=state.stage===6?'Structured graph query':state.stage===5?'Recognition evidence':'Pipeline state';ui['data-frame'].textContent=state.stage<4?`view ${viewCount()}`:'robot_base';const displayData=state.stage===4?{instances:objects.map(o=>({id:o.instance,surface_samples:allPoints.filter(p=>p.object.id===o.id).length})),frame:'robot_base'}:step.data;ui['data-preview'].textContent=JSON.stringify(displayData,null,2);ui['stage-takeaway'].textContent=step.takeaway;const v=state.mode==='single'?sensorViews[0]:sensorViews[Math.min(state.stage,3)];ui['view-status'].textContent=`${viewCount()} view${viewCount()===1?'':'s'} · ${v.name}`;ui['point-status'].textContent=`${measuredPoints().length.toLocaleString()} simulated samples`;ui['map-status'].textContent=state.stage<3?`${coverage()}% visible surface`:state.stage===3?'Fused local 3D map':state.stage===4?'3 object instances':state.stage===5?'cup · cube · bottle':state.stage===6?'Target: cup_1':'Cup moved · map refreshed';}

  function resize(){const r=canvas.getBoundingClientRect(),dpr=Math.min(devicePixelRatio||1,2);canvas.width=Math.round(r.width*dpr);canvas.height=Math.round(r.height*dpr);ctx.setTransform(dpr,0,0,dpr,0,0);draw();}
  function project(p){const r=canvas.getBoundingClientRect(),cy=Math.cos(state.yaw),sy=Math.sin(state.yaw),cp=Math.cos(state.pitch),sp=Math.sin(state.pitch);const x1=cy*p[0]-sy*p[1],y1=sy*p[0]+cy*p[1],y2=cp*p[2]-sp*y1,z2=sp*p[2]+cp*y1,scale=Math.min(r.width,r.height)*.41*state.zoom;return{x:r.width*.5+x1*scale,y:r.height*.62-y2*scale,depth:z2};}
  function line3(a,b,color,width=1,dash=[]){const p=project(a),q=project(b);ctx.save();ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash(dash);ctx.beginPath();ctx.moveTo(p.x,p.y);ctx.lineTo(q.x,q.y);ctx.stroke();ctx.restore();}
  function poly3(points,color,width=1,dash=[]){ctx.save();ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash(dash);ctx.beginPath();points.forEach((x,i)=>{const p=project(x);if(i)ctx.lineTo(p.x,p.y);else ctx.moveTo(p.x,p.y);});ctx.stroke();ctx.restore();}
  function label3(p,text,color,offset=[0,0],bold=false){const q=project(p);ctx.fillStyle=color;ctx.font=`${bold?'600 ':''}12px ui-monospace, SFMono-Regular, Menlo, monospace`;ctx.fillText(text,q.x+offset[0],q.y+offset[1]);}
  function drawTable(){const min=-.72,max=.72,step=.18;for(let v=min;v<=max+.001;v+=step){line3([v,min,0],[v,max,0],'#223847',.7);line3([min,v,0],[max,v,0],'#223847',.7);}const ring=[[min,min],[max,min],[max,max],[min,max],[min,min]];for(let i=0;i<4;i++)line3([ring[i][0],ring[i][1],0],[ring[i+1][0],ring[i+1][1],0],'#688297',1);}
  function boxCorners(o,offset=[0,0,0]){const[sx,sy,sz]=o.size.map(v=>v/2),[cx,cy,cz]=add(o.center,offset);return[[-sx,-sy,-sz],[sx,-sy,-sz],[sx,sy,-sz],[-sx,sy,-sz],[-sx,-sy,sz],[sx,-sy,sz],[sx,sy,sz],[-sx,sy,sz]].map(v=>[cx+v[0],cy+v[1],cz+v[2]]);}
  function boxOutline(o,color,width=1,dash=[],offset=[0,0,0]){const c=boxCorners(o,offset);boxEdges.forEach(e=>line3(c[e[0]],c[e[1]],color,width,dash));}
  function circlePoints(center,r,z,count=40){const p=[];for(let i=0;i<=count;i++){const t=2*Math.PI*i/count;p.push([center[0]+r*Math.cos(t),center[1]+r*Math.sin(t),z]);}return p;}
  function objectOutline(o,color,width=1,dash=[],offset=[0,0,0]){
    const c=add(o.center,offset);
    if(o.kind==='cube'){boxOutline(o,color,width,dash,offset);return;}
    if(o.kind==='cup'){
      poly3(circlePoints(c,.095,c[2]-.09),color,width,dash);poly3(circlePoints(c,.095,c[2]+.09),color,width,dash);
      [0,Math.PI/2,Math.PI,3*Math.PI/2].forEach(t=>line3([c[0]+.095*Math.cos(t),c[1]+.095*Math.sin(t),c[2]-.09],[c[0]+.095*Math.cos(t),c[1]+.095*Math.sin(t),c[2]+.09],color,width,dash));
      const h=[];for(let i=0;i<=32;i++){const t=2*Math.PI*i/32;h.push([c[0]-.125+.062*Math.cos(t),c[1],c[2]+.005+.065*Math.sin(t)]);}poly3(h,color,width,dash);return;
    }
    poly3(circlePoints(c,.078,.005),color,width,dash);poly3(circlePoints(c,.078,.19),color,width,dash);poly3(circlePoints(c,.034,.225),color,width,dash);poly3(circlePoints(c,.034,.298),color,width,dash);
    [0,Math.PI/2,Math.PI,3*Math.PI/2].forEach(t=>{line3([c[0]+.078*Math.cos(t),c[1]+.078*Math.sin(t),.005],[c[0]+.078*Math.cos(t),c[1]+.078*Math.sin(t),.19],color,width,dash);line3([c[0]+.078*Math.cos(t),c[1]+.078*Math.sin(t),.19],[c[0]+.034*Math.cos(t),c[1]+.034*Math.sin(t),.225],color,width,dash);line3([c[0]+.034*Math.cos(t),c[1]+.034*Math.sin(t),.225],[c[0]+.034*Math.cos(t),c[1]+.034*Math.sin(t),.298],color,width,dash);});
  }
  function transformedPoint(pt){return state.stage===7&&pt.object.id==='cup_1'?add(pt.p,cupOffset()):pt.p;}
  function drawPoints(){measuredPoints().map(pt=>({pt,q:project(transformedPoint(pt))})).sort((a,b)=>a.q.depth-b.q.depth).forEach(({pt,q})=>{ctx.globalAlpha=state.stage<4?.92:.48;ctx.fillStyle=pt.object.color;ctx.beginPath();ctx.arc(q.x,q.y,state.stage<3?1.55:1.3,0,Math.PI*2);ctx.fill();});ctx.globalAlpha=1;}
  function currentSensor(){return state.mode==='single'?sensorViews[0]:sensorViews[Math.min(state.stage,3)];}
  function drawSensor(){if(state.stage>3)return;const s=currentSensor(),p=project(s.pos);line3(s.pos,[0,0,.05],'#ffac4d',1.5,[5,4]);ctx.save();ctx.translate(p.x,p.y);ctx.rotate(-.18);ctx.fillStyle='#ffac4d';ctx.strokeStyle='#ecf5fb';ctx.lineWidth=1;ctx.beginPath();ctx.roundRect(-15,-9,30,18,4);ctx.fill();ctx.stroke();ctx.beginPath();ctx.arc(0,0,4,0,Math.PI*2);ctx.fillStyle='#071018';ctx.fill();ctx.restore();label3(s.pos,s.name,'#ecf5fb',[18,-10]);}
  function drawInstances(){if(state.stage<4)return;objects.forEach(o=>{const offset=state.stage===7&&o.id==='cup_1'?cupOffset():[0,0,0];const selected=state.stage>=6&&o.id==='cup_1',color=selected?'#64e3a1':'#d7e8f2';objectOutline(o,color,selected?3:1.4,selected?[]:[4,3],offset);const label=state.stage===4?o.instance:state.stage>=5?`${o.label} · ${Math.round(o.score*100)}%`:o.id;label3(add(add(o.center,offset),[0,0,o.size[2]/2+.055]),label,color,[-34,0],selected);});}
  function arrow3(a,b,color,label){const p=project(a),q=project(b),angle=Math.atan2(q.y-p.y,q.x-p.x);ctx.save();ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(p.x,p.y);ctx.lineTo(q.x,q.y);ctx.stroke();ctx.beginPath();ctx.moveTo(q.x,q.y);ctx.lineTo(q.x-9*Math.cos(angle-.45),q.y-9*Math.sin(angle-.45));ctx.lineTo(q.x-9*Math.cos(angle+.45),q.y-9*Math.sin(angle+.45));ctx.closePath();ctx.fill();ctx.font='12px ui-monospace,monospace';ctx.fillText(label,(p.x+q.x)/2+5,(p.y+q.y)/2-5);ctx.restore();}
  function drawGraph(){if(state.stage!==6)return;arrow3(add(objects[0].center,[0,0,.20]),add(objects[1].center,[0,0,.20]),'#b58cff','left_of');arrow3(add(objects[2].center,[0,0,.24]),add(objects[1].center,[0,0,.24]),'#b58cff','behind');}
  function drawRobot(){if(state.stage!==7)return;const joints=[[-.68,.55,0],[-.55,.45,.28],[-.33,.32,.48],[-.12,.20,.39]];for(let i=0;i<joints.length-1;i++)line3(joints[i],joints[i+1],'#d7e8f2',5);joints.forEach(j=>{const p=project(j);ctx.fillStyle='#071018';ctx.strokeStyle='#d7e8f2';ctx.lineWidth=2;ctx.beginPath();ctx.arc(p.x,p.y,5,0,Math.PI*2);ctx.fill();ctx.stroke();});const start=objects[0].center,dest=add(start,[.72,-.04,0]),path=[joints.at(-1),add(start,[0,0,.28]),add(start,[0,0,.10]),add(dest,[0,0,.28]),add(dest,[0,0,.10])];for(let i=0;i<path.length-1;i++)line3(path[i],path[i+1],'#b58cff',2,[7,5]);const p=project(add(objects[0].center,cupOffset()));ctx.fillStyle='#64e3a1';ctx.beginPath();ctx.arc(p.x,p.y,6,0,Math.PI*2);ctx.fill();label3(add(dest,[0,0,.03]),'right_region','#b58cff',[10,0]);}
  function draw(){const r=canvas.getBoundingClientRect();if(!r.width)return;ctx.clearRect(0,0,r.width,r.height);const g=ctx.createRadialGradient(r.width*.48,r.height*.48,20,r.width*.48,r.height*.48,r.width*.7);g.addColorStop(0,'#102331');g.addColorStop(1,'#08131c');ctx.fillStyle=g;ctx.fillRect(0,0,r.width,r.height);drawTable();drawPoints();drawSensor();if(state.stage<4)objects.forEach(o=>objectOutline(o,'#294454',.8,[3,4]));drawInstances();drawGraph();drawRobot();if(state.selected){const o=objects.find(x=>x.id===state.selected);if(o)objectOutline(o,'#ffdc69',3);}}
  function hitObject(x,y){let best=null,d=38;objects.forEach(o=>{const offset=state.stage===7&&o.id==='cup_1'?cupOffset():[0,0,0],p=project(add(o.center,offset)),v=Math.hypot(x-p.x,y-p.y);if(v<d){best=o;d=v;}});return best;}

  let timer=null;
  function stop(){state.playing=false;playButton.textContent='▶ Play route';if(timer){clearInterval(timer);timer=null;}}
  function play(){state.playing=true;playButton.textContent='Ⅱ Pause';if(state.stage===stages.length-1)setStage(0);timer=setInterval(()=>{if(state.stage===stages.length-1){stop();return;}setStage(state.stage+1);},2200);}
  playButton.addEventListener('click',()=>state.playing?stop():play());
  document.getElementById('previous-button').addEventListener('click',()=>{stop();setStage(state.stage-1);});
  document.getElementById('next-button').addEventListener('click',()=>{stop();setStage(state.stage+1);});
  document.getElementById('reset-view').addEventListener('click',()=>{state.yaw=-.72;state.pitch=.72;state.zoom=1;draw();});
  modeSelect.addEventListener('change',()=>{state.mode=modeSelect.value;stop();updatePanel();draw();});
  canvas.addEventListener('pointerdown',e=>{canvas.setPointerCapture(e.pointerId);state.drag={x:e.clientX,y:e.clientY,yaw:state.yaw,pitch:state.pitch,moved:false};});
  canvas.addEventListener('pointermove',e=>{if(!state.drag)return;const dx=e.clientX-state.drag.x,dy=e.clientY-state.drag.y;if(Math.abs(dx)+Math.abs(dy)>3)state.drag.moved=true;state.yaw=state.drag.yaw+dx*.008;state.pitch=Math.max(.18,Math.min(1.35,state.drag.pitch+dy*.006));draw();});
  canvas.addEventListener('pointerup',e=>{if(state.drag&&!state.drag.moved){const r=canvas.getBoundingClientRect(),o=hitObject(e.clientX-r.left,e.clientY-r.top);state.selected=o?o.id:null;if(o){ui['data-label'].textContent=`Selected · ${o.label}`;ui['data-preview'].textContent=JSON.stringify({id:o.id,category:o.label,center_base_m:o.center,dimensions_m:o.size,geometry_evidence:o.evidence,semantic_score:o.score,observation:state.stage>=3?'multi_view_fused':'partial_surface'},null,2);}else updatePanel();draw();}state.drag=null;});
  canvas.addEventListener('wheel',e=>{e.preventDefault();state.zoom=Math.max(.68,Math.min(1.6,state.zoom*Math.exp(-e.deltaY*.001)));draw();},{passive:false});
  new ResizeObserver(resize).observe(canvas);
  function animate(){state.phase+=.018;if(state.stage===7)draw();requestAnimationFrame(animate);}
  buildStepper();setStage(0);resize();animate();
})();
