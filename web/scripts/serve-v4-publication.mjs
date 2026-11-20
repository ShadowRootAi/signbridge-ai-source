/** Loopback preview of the exact prepared artifact, with byte-range MP4 support. */
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const web=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const root=path.resolve(web,'../publication-v4/dist');
const headers=JSON.parse(await readFile(path.join(web,'deployment/headers.json'),'utf8'));
const mime={'.html':'text/html; charset=utf-8','.js':'text/javascript','.mjs':'text/javascript','.css':'text/css','.json':'application/json','.txt':'text/plain; charset=utf-8','.wasm':'application/wasm','.mp4':'video/mp4','.svg':'image/svg+xml'};
const server=createServer(async(req,res)=>{
  for(const [key,value] of Object.entries(headers)) res.setHeader(key,value);
  if(!['GET','HEAD'].includes(req.method??'')){res.writeHead(405,{Allow:'GET, HEAD'});res.end();return;}
  try {
    const route=decodeURIComponent(new URL(req.url,'http://127.0.0.1:5176').pathname);
    if(route==='/_headers'||route.startsWith('/__signbridge/')||route.startsWith('/@fs/')){res.writeHead(404);res.end();return;}
    const target=path.resolve(root,`.${route==='/'?'/index.html':route}`),relative=path.relative(root,target);
    if(relative.startsWith('..')||path.isAbsolute(relative)){res.writeHead(403);res.end();return;}
    const body=await readFile(target), type=mime[path.extname(target)]??'application/octet-stream';
    const range=req.headers.range;
    if(range){
      const parsed=/^bytes=(\d*)-(\d*)$/.exec(range);
      if(!parsed||(!parsed[1]&&!parsed[2])){res.writeHead(416,{'Content-Range':`bytes */${body.length}`});res.end();return;}
      const start=parsed[1]?Number(parsed[1]):Math.max(0,body.length-Number(parsed[2]));
      const end=parsed[1]?(parsed[2]?Math.min(Number(parsed[2]),body.length-1):body.length-1):body.length-1;
      if(start>end||start>=body.length){res.writeHead(416,{'Content-Range':`bytes */${body.length}`});res.end();return;}
      const bytes=body.subarray(start,end+1);
      res.writeHead(206,{'Content-Type':type,'Content-Range':`bytes ${start}-${end}/${body.length}`,'Content-Length':bytes.length,'Accept-Ranges':'bytes'});
      res.end(req.method==='HEAD'?undefined:bytes);return;
    }
    res.writeHead(200,{'Content-Type':type,'Content-Length':body.length,'Accept-Ranges':'bytes'});res.end(req.method==='HEAD'?undefined:body);
  }catch{res.writeHead(404);res.end();}
});
server.listen(5176,'127.0.0.1',()=>console.log('SignBridge V4 review: http://127.0.0.1:5176'));
