(function(){var e=Object.create,t=Object.defineProperty,n=Object.getOwnPropertyDescriptor,r=Object.getOwnPropertyNames,i=Object.getPrototypeOf,a=Object.prototype.hasOwnProperty,o=(e,t)=>()=>(t||(e((t={exports:{}}).exports,t),e=null),t.exports),s=(e,i,o,s)=>{if(i&&typeof i==`object`||typeof i==`function`)for(var c=r(i),l=0,u=c.length,d;l<u;l++)d=c[l],!a.call(e,d)&&d!==o&&t(e,d,{get:(e=>i[e]).bind(null,d),enumerable:!(s=n(i,d))||s.enumerable});return e},c=(n,r,o)=>(o=n==null?{}:e(i(n)),s(r||!n||!n.__esModule||!a.call(n,`default`)?t(o,`default`,{value:n,enumerable:!0}):o,n)),l=class{constructor(e,t){
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
this.backend=e,this.dataMover=t,this.data=new WeakMap,this.dataIdsCount=0}get(e){return this.data.has(e)||this.dataMover.moveData(this.backend,e),this.data.get(e)}set(e,t){this.dataIdsCount++,this.data.set(e,t)}has(e){return this.data.has(e)}delete(e){return this.dataIdsCount--,this.data.delete(e)}numDataIds(){return this.dataIdsCount}},u=class{refCount(e){return d(`refCount`)}incRef(e){return d(`incRef`)}timerAvailable(){return!0}time(e){return d(`time`)}read(e){return d(`read`)}readSync(e){return d(`readSync`)}readToGPU(e,t){return d(`readToGPU`)}numDataIds(){return d(`numDataIds`)}disposeData(e,t){return d(`disposeData`)}write(e,t,n){return d(`write`)}move(e,t,n,r,i){return d(`move`)}createTensorFromGPUData(e,t,n){return d(`createTensorFromGPUData`)}memory(){return d(`memory`)}floatPrecision(){return d(`floatPrecision`)}epsilon(){return this.floatPrecision()===32?1e-7:1e-4}dispose(){return d(`dispose`)}};function d(e){throw Error(`'${e}' not yet implemented or not found in the registry. This kernel may not be supported by the tfjs backend you have chosen`)}
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function f(e,t,n){return Math.max(e,Math.min(t,n))}function p(e,t,n){let r=e[t];e[t]=e[n],e[n]=r}function m(e,t){if(!e)throw Error(typeof t==`string`?t:t())}function h(e,t,n=``){m(v(e,t),()=>n+` Shapes ${e} and ${t} must match`)}function g(e){m(e!=null,()=>`The input to the tensor constructor must be a non-null value.`)}function _(e){if(e.length===0)return 1;let t=e[0];for(let n=1;n<e.length;n++)t*=e[n];return t}function v(e,t){if(e===t)return!0;if(e==null||t==null||e.length!==t.length)return!1;for(let n=0;n<e.length;n++)if(e[n]!==t[n])return!1;return!0}function y(e){return e%1==0}function b(e,t){return t<=e.length?e:e+` `.repeat(t-e.length)}function x(e,t){let n=1,r=-1;for(let t=0;t<e.length;++t)if(e[t]>=0)n*=e[t];else if(e[t]===-1){if(r!==-1)throw Error(`Shapes can only have 1 implicit size. Found -1 at dim ${r} and dim ${t}`);r=t}else if(e[t]<0)throw Error(`Shapes can not be < 0. Found ${e[t]} at dim ${t}`);if(r===-1){if(t>0&&t!==n)throw Error(`Size(${t}) must match the product of shape ${e}`);return e}if(n===0)throw Error(`Cannot infer the missing size in [${e}] when there are 0 elements`);if(t%n!==0)throw Error(`The implicit shape can't be a fractional number. Got ${t} / ${n}`);let i=e.slice();return i[r]=t/n,i}function S(e,t){let n=t.length;return e=e==null?t.map((e,t)=>t):[].concat(e),m(e.every(e=>e>=-n&&e<n),()=>`All values in axis param must be in range [-${n}, ${n}) but got axis ${e}`),m(e.every(e=>y(e)),()=>`All values in axis param must be integers but got axis ${e}`),e.map(e=>e<0?n+e:e)}function C(e,t){return w(e,t)}function w(e,t){let n=null;if(e==null||e===`float32`)n=new Float32Array(t);else if(e===`int32`)n=new Int32Array(t);else if(e===`bool`)n=new Uint8Array(t);else if(e===`string`)n=Array(t);else throw Error(`Unknown data type ${e}`);return n}function T(e,t){for(let n=0;n<e.length;n++){let r=e[n];if(isNaN(r)||!isFinite(r))throw Error(`A tensor of type ${t} being uploaded contains ${r}.`)}}function E(e){return e===`bool`||e===`complex64`||e===`float32`||e===`int32`||e===`string`}function D(e,t){return!(t===`complex64`||t===`float32`&&e!==`complex64`||t===`int32`&&e!==`float32`&&e!==`complex64`||t===`bool`&&e===`bool`)}function O(e){if(e===`float32`||e===`int32`)return 4;if(e===`complex64`)return 8;if(e===`bool`)return 1;throw Error(`Unknown dtype ${e}`)}function k(e){if(e==null)return 0;let t=0;return e.forEach(e=>t+=e.length),t}function A(e){return typeof e==`string`||e instanceof String}function j(e){return typeof e==`boolean`}function M(e){return typeof e==`number`}function N(e){return Array.isArray(e)?N(e[0]):e instanceof Float32Array?`float32`:e instanceof Int32Array||e instanceof Uint8Array||e instanceof Uint8ClampedArray?`int32`:M(e)?`float32`:A(e)?`string`:j(e)?`bool`:`float32`}function ee(e){return!!(e&&e.constructor&&e.call&&e.apply)}function P(e){let t=e.length;if(t<2)return[];let n=Array(t-1);n[t-2]=e[t-1];for(let r=t-3;r>=0;--r)n[r]=n[r+1]*e[r+1];return n}function te(e,t,n,r=!1){let i=[];if(t.length===1){let a=t[0]*(r?2:1);for(let t=0;t<a;t++)i[t]=n[e+t]}else{let a=t[0],o=t.slice(1),s=o.reduce((e,t)=>e*t)*(r?2:1);for(let t=0;t<a;t++)i[t]=te(e+t*s,o,n,r)}return i}function F(e,t,n=!1){if(e.length===0)return t[0];let r=e.reduce((e,t)=>e*t)*(n?2:1);if(r===0)return[];if(r!==t.length)throw Error(`[${e}] does not match the input size ${t.length}${n?` for a complex tensor`:``}.`);return te(0,e,t,n)}function ne(e,t){if(Array.isArray(e))return e;if(t===`float32`)return e instanceof Float32Array?e:new Float32Array(e);if(t===`int32`)return e instanceof Int32Array?e:new Int32Array(e);if(t===`bool`||t===`string`)return Uint8Array.from(new Int32Array(e));throw Error(`Unknown dtype ${t}`)}function re(e,t){let n=I(e,t);for(let e=0;e<n.length;e++)n[e]=1;return n}function I(e,t){if(t==null||t===`float32`||t===`complex64`)return new Float32Array(e);if(t===`int32`)return new Int32Array(e);if(t===`bool`)return new Uint8Array(e);throw Error(`Unknown data type ${t}`)}function ie(e,t){let n=e.reduce((e,t)=>e*t,1);if(t==null||t===`float32`)return F(e,new Float32Array(n));if(t===`int32`)return F(e,new Int32Array(n));if(t===`bool`)return F(e,new Uint8Array(n));throw Error(`Unknown data type ${t}`)}function L(e){e.forEach(t=>{m(Number.isInteger(t)&&t>=0,()=>`Tensor must have a shape comprised of positive integers but got shape [${e}].`)})}function R(e,t,n){if(t===0)return 0;if(t===1)return e[0];let r=e[e.length-1];for(let t=0;t<e.length-1;++t)r+=n[t]*e[t];return r}function ae(e,t,n){if(t===0)return[];if(t===1)return[e];let r=Array(t);for(let t=0;t<r.length-1;++t)r[t]=Math.floor(e/n[t]),e-=r[t]*n[t];return r[r.length-1]=e,r}function oe(e){return e&&e.then&&typeof e.then==`function`}
/**
* @license
* Copyright 2017 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let se=`tfjsflags`;var ce=class{constructor(e){this.global=e,this.flags={},this.flagRegistry={},this.urlFlags={},this.getQueryParams=le,this.populateURLFlags()}setPlatform(e,t){this.platform!=null&&(z().getBool(`IS_TEST`)||z().getBool(`PROD`)||console.warn(`Platform ${this.platformName} has already been set. Overwriting the platform with ${e}.`)),this.platformName=e,this.platform=t}registerFlag(e,t,n){if(this.flagRegistry[e]={evaluationFn:t,setHook:n},this.urlFlags[e]!=null){let t=this.urlFlags[e];z().getBool(`IS_TEST`)||z().getBool(`PROD`)||console.warn(`Setting feature override from URL ${e}: ${t}.`),this.set(e,t)}}async getAsync(e){return e in this.flags||(this.flags[e]=await this.evaluateFlag(e)),this.flags[e]}get(e){if(e in this.flags)return this.flags[e];let t=this.evaluateFlag(e);if(oe(t))throw Error(`Flag ${e} cannot be synchronously evaluated. Please use getAsync() instead.`);return this.flags[e]=t,this.flags[e]}getNumber(e){return this.get(e)}getBool(e){return this.get(e)}getString(e){return this.get(e)}getFlags(){return this.flags}get features(){return this.flags}set(e,t){if(this.flagRegistry[e]==null)throw Error(`Cannot set flag ${e} as it has not been registered.`);this.flags[e]=t,this.flagRegistry[e].setHook!=null&&this.flagRegistry[e].setHook(t)}evaluateFlag(e){if(this.flagRegistry[e]==null)throw Error(`Cannot evaluate flag '${e}': no evaluation function found.`);return this.flagRegistry[e].evaluationFn()}setFlags(e){this.flags=Object.assign({},e)}reset(){this.flags={},this.urlFlags={},this.populateURLFlags()}populateURLFlags(){if(this.global===void 0||this.global.location===void 0||this.global.location.search===void 0)return;let e=this.getQueryParams(this.global.location.search);se in e&&e[se].split(`,`).forEach(e=>{let[t,n]=e.split(`:`);this.urlFlags[t]=de(t,n)})}};function le(e){let t={};return e.replace(/[?&]([^=?&]+)(?:=([^&]*))?/g,(e,...n)=>(ue(t,n[0],n[1]),n.join(`=`))),t}function ue(e,t,n){e[decodeURIComponent(t)]=decodeURIComponent(n||``)}function de(e,t){let n=t.toLowerCase();return n===`true`||n===`false`?n===`true`:`${+n}`===n?+n:t}function z(){return fe}let fe=null;function pe(e){fe=e}
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let me;function he(){if(me==null){let e;if(typeof window<`u`)e=window;else if(typeof global<`u`)e=global;else if(typeof process<`u`)e=process;else if(typeof self<`u`)e=self;else throw Error(`Could not find a global object`);me=e}return me}function ge(){let e=he();return e._tfGlobals??=new Map,e._tfGlobals}function _e(e,t){let n=ge();if(n.has(e))return n.get(e);{let r=t();return n.set(e,r),n.get(e)}}let ve=`Acos`,ye=`Acosh`,be=`AddN`,xe=`ArgMax`,Se=`ArgMin`,Ce=`Asin`,we=`Asinh`,Te=`Atan`,Ee=`Atanh`,De=`Atan2`,Oe=`AvgPool`,ke=`AvgPoolGrad`,Ae=`AvgPool3D`,je=`AvgPool3DGrad`,Me=`BatchMatMul`,Ne=`BatchToSpaceND`,Pe=`Bincount`,Fe=`BitwiseAnd`,Ie=`BroadcastArgs`,Le=`Cast`,Re=`Ceil`,ze=`ClipByValue`,Be=`ComplexAbs`,Ve=`Concat`,He=`Conv2D`,Ue=`Conv2DBackpropInput`,We=`Conv3D`,B=`Conv3DBackpropFilterV2`,Ge=`Conv3DBackpropInputV2`,Ke=`Cosh`,qe=`Cumprod`,Je=`Cumsum`,Ye=`CropAndResize`,Xe=`DenseBincount`,Ze=`DepthToSpace`,Qe=`DepthwiseConv2dNative`,$e=`Diag`,et=`Dilation2D`,tt=`Dilation2DBackpropInput`,nt=`Dilation2DBackpropFilter`,rt=`RealDiv`,it=`EluGrad`,at=`Equal`,ot=`ExpandDims`,st=`Expm1`,ct=`Fill`,lt=`FlipLeftRight`,ut=`Floor`,dt=`FloorDiv`,ft=`FusedBatchNorm`,pt=`GatherV2`,mt=`GatherNd`,ht=`Greater`,gt=`GreaterEqual`,_t=`Identity`,vt=`IsFinite`,yt=`IsInf`,bt=`IsNan`,xt=`LeakyRelu`,St=`Less`,Ct=`LessEqual`,wt=`LinSpace`,Tt=`Log1p`,Et=`LogicalAnd`,Dt=`LogicalNot`,Ot=`LogicalOr`,kt=`LRNGrad`,At=`Maximum`,jt=`MaxPool`,Mt=`MaxPoolGrad`,Nt=`MaxPool3D`,Pt=`MaxPool3DGrad`,Ft=`MaxPoolWithArgmax`,It=`Mean`,Lt=`Minimum`,Rt=`MirrorPad`,zt=`Multinomial`,Bt=`Multiply`,Vt=`NotEqual`,Ht=`NonMaxSuppressionV3`,Ut=`NonMaxSuppressionV4`,Wt=`NonMaxSuppressionV5`,Gt=`OnesLike`,Kt=`OneHot`,qt=`Pack`,Jt=`PadV2`,Yt=`Prelu`,Xt=`Prod`,Zt=`Range`,Qt=`Reciprocal`,$t=`Relu`,en=`Reshape`,tn=`ResizeNearestNeighbor`,nn=`ResizeNearestNeighborGrad`,rn=`ResizeBilinear`,an=`ResizeBilinearGrad`,on=`Relu6`,sn=`Reverse`,cn=`Round`,ln=`Rsqrt`,un=`ScatterNd`,dn=`TensorScatterUpdate`,fn=`SearchSorted`,pn=`Select`,mn=`Selu`,hn=`Slice`,gn=`Sinh`,_n=`Sign`,vn=`Sigmoid`,yn=`Softplus`,bn=`Sqrt`,xn=`SpaceToBatchND`,Sn=`SplitV`,Cn=`Softmax`,wn=`SparseFillEmptyRows`,Tn=`SparseReshape`,En=`SparseSegmentMean`,Dn=`SparseSegmentSum`,On=`SparseToDense`,kn=`SquaredDifference`,An=`Square`,jn=`StaticRegexReplace`,Mn=`StridedSlice`,Nn=`StringNGrams`,Pn=`StringSplit`,Fn=`StringToHashBucketFast`,In=`Tanh`,Ln=`Tile`,Rn=`TopK`,zn=`Transform`,Bn=`Transpose`,Vn=`Unique`,Hn=`Unpack`,Un=`ZerosLike`,Wn=`Step`,Gn=`RotateWithOffset`,Kn=`_FusedMatMul`,qn=`FusedConv2D`,Jn=`FusedDepthwiseConv2D`;
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Yn(...e){z().getBool(`IS_TEST`)||z().getBool(`PROD`)||console.warn(...e)}
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let Xn=_e(`kernelRegistry`,()=>new Map),Zn=_e(`gradRegistry`,()=>new Map);function Qn(e,t){let n=nr(e,t);return Xn.get(n)}function $n(e){return Zn.get(e)}function er(e){let t=Xn.entries(),n=[];for(;;){let{done:r,value:i}=t.next();if(r)break;let[a,o]=i,[s]=a.split(`_`);s===e&&n.push(o)}return n}function tr(e){let{kernelName:t,backendName:n}=e,r=nr(t,n);Xn.has(r)&&Yn(`The kernel '${t}' for backend '${n}' is already registered`),Xn.set(r,e)}function nr(e,t){return`${t}_${e}`}
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function rr(e){return e instanceof Float32Array||e instanceof Int32Array||e instanceof Uint8Array||e instanceof Uint8ClampedArray}var ir=c(o(((e,t)=>{t.exports=r;var n=null;try{n=new WebAssembly.Instance(new WebAssembly.Module(new Uint8Array([0,97,115,109,1,0,0,0,1,13,2,96,0,1,127,96,4,127,127,127,127,1,127,3,7,6,0,1,1,1,1,1,6,6,1,127,1,65,0,11,7,50,6,3,109,117,108,0,1,5,100,105,118,95,115,0,2,5,100,105,118,95,117,0,3,5,114,101,109,95,115,0,4,5,114,101,109,95,117,0,5,8,103,101,116,95,104,105,103,104,0,0,10,191,1,6,4,0,35,0,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,126,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,127,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,128,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,129,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,130,34,4,66,32,135,167,36,0,32,4,167,11])),{}).exports}catch{}function r(e,t,n){this.low=e|0,this.high=t|0,this.unsigned=!!n}r.prototype.__isLong__,Object.defineProperty(r.prototype,"__isLong__",{value:!0});function i(e){return(e&&e.__isLong__)===!0}r.isLong=i;var a={},o={};function s(e,t){var n,r,i;return t?(e>>>=0,(i=0<=e&&e<256)&&(r=o[e],r)?r:(n=l(e,(e|0)<0?-1:0,!0),i&&(o[e]=n),n)):(e|=0,(i=-128<=e&&e<128)&&(r=a[e],r)?r:(n=l(e,e<0?-1:0,!1),i&&(a[e]=n),n))}r.fromInt=s;function c(e,t){if(isNaN(e))return t?b:y;if(t){if(e<0)return b;if(e>=g)return T}else{if(e<=-_)return E;if(e+1>=_)return w}return e<0?c(-e,t).neg():l(e%h|0,e/h|0,t)}r.fromNumber=c;function l(e,t,n){return new r(e,t,n)}r.fromBits=l;var u=Math.pow;function d(e,t,n){if(e.length===0)throw Error(`empty string`);if(e===`NaN`||e===`Infinity`||e===`+Infinity`||e===`-Infinity`)return y;if(typeof t==`number`?(n=t,t=!1):t=!!t,n||=10,n<2||36<n)throw RangeError(`radix`);var r;if((r=e.indexOf(`-`))>0)throw Error(`interior hyphen`);if(r===0)return d(e.substring(1),t,n).neg();for(var i=c(u(n,8)),a=y,o=0;o<e.length;o+=8){var s=Math.min(8,e.length-o),l=parseInt(e.substring(o,o+s),n);if(s<8){var f=c(u(n,s));a=a.mul(f).add(c(l))}else a=a.mul(i),a=a.add(c(l))}return a.unsigned=t,a}r.fromString=d;function f(e,t){return typeof e==`number`?c(e,t):typeof e==`string`?d(e,t):l(e.low,e.high,typeof t==`boolean`?t:e.unsigned)}r.fromValue=f;var p=65536,m=1<<24,h=p*p,g=h*h,_=g/2,v=s(m),y=s(0);r.ZERO=y;var b=s(0,!0);r.UZERO=b;var x=s(1);r.ONE=x;var S=s(1,!0);r.UONE=S;var C=s(-1);r.NEG_ONE=C;var w=l(-1,2147483647,!1);r.MAX_VALUE=w;var T=l(-1,-1,!0);r.MAX_UNSIGNED_VALUE=T;var E=l(0,-2147483648,!1);r.MIN_VALUE=E;var D=r.prototype;D.toInt=function(){return this.unsigned?this.low>>>0:this.low},D.toNumber=function(){return this.unsigned?(this.high>>>0)*h+(this.low>>>0):this.high*h+(this.low>>>0)},D.toString=function(e){if(e||=10,e<2||36<e)throw RangeError(`radix`);if(this.isZero())return`0`;if(this.isNegative()){if(this.eq(E)){var t=c(e),n=this.div(t),r=n.mul(t).sub(this);return n.toString(e)+r.toInt().toString(e)}return`-`+this.neg().toString(e)}for(var i=c(u(e,6),this.unsigned),a=this,o=``;;){var s=a.div(i),l=(a.sub(s.mul(i)).toInt()>>>0).toString(e);if(a=s,a.isZero())return l+o;for(;l.length<6;)l=`0`+l;o=``+l+o}},D.getHighBits=function(){return this.high},D.getHighBitsUnsigned=function(){return this.high>>>0},D.getLowBits=function(){return this.low},D.getLowBitsUnsigned=function(){return this.low>>>0},D.getNumBitsAbs=function(){if(this.isNegative())return this.eq(E)?64:this.neg().getNumBitsAbs();for(var e=this.high==0?this.low:this.high,t=31;t>0&&!(e&1<<t);t--);return this.high==0?t+1:t+33},D.isZero=function(){return this.high===0&&this.low===0},D.eqz=D.isZero,D.isNegative=function(){return!this.unsigned&&this.high<0},D.isPositive=function(){return this.unsigned||this.high>=0},D.isOdd=function(){return(this.low&1)==1},D.isEven=function(){return!(this.low&1)},D.equals=function(e){return i(e)||(e=f(e)),this.unsigned!==e.unsigned&&this.high>>>31==1&&e.high>>>31==1?!1:this.high===e.high&&this.low===e.low},D.eq=D.equals,D.notEquals=function(e){return!this.eq(e)},D.neq=D.notEquals,D.ne=D.notEquals,D.lessThan=function(e){return this.comp(e)<0},D.lt=D.lessThan,D.lessThanOrEqual=function(e){return this.comp(e)<=0},D.lte=D.lessThanOrEqual,D.le=D.lessThanOrEqual,D.greaterThan=function(e){return this.comp(e)>0},D.gt=D.greaterThan,D.greaterThanOrEqual=function(e){return this.comp(e)>=0},D.gte=D.greaterThanOrEqual,D.ge=D.greaterThanOrEqual,D.compare=function(e){if(i(e)||(e=f(e)),this.eq(e))return 0;var t=this.isNegative(),n=e.isNegative();return t&&!n?-1:!t&&n?1:this.unsigned?e.high>>>0>this.high>>>0||e.high===this.high&&e.low>>>0>this.low>>>0?-1:1:this.sub(e).isNegative()?-1:1},D.comp=D.compare,D.negate=function(){return!this.unsigned&&this.eq(E)?E:this.not().add(x)},D.neg=D.negate,D.add=function(e){i(e)||(e=f(e));var t=this.high>>>16,n=this.high&65535,r=this.low>>>16,a=this.low&65535,o=e.high>>>16,s=e.high&65535,c=e.low>>>16,u=e.low&65535,d=0,p=0,m=0,h=0;return h+=a+u,m+=h>>>16,h&=65535,m+=r+c,p+=m>>>16,m&=65535,p+=n+s,d+=p>>>16,p&=65535,d+=t+o,d&=65535,l(m<<16|h,d<<16|p,this.unsigned)},D.subtract=function(e){return i(e)||(e=f(e)),this.add(e.neg())},D.sub=D.subtract,D.multiply=function(e){if(this.isZero())return y;if(i(e)||(e=f(e)),n)return l(n.mul(this.low,this.high,e.low,e.high),n.get_high(),this.unsigned);if(e.isZero())return y;if(this.eq(E))return e.isOdd()?E:y;if(e.eq(E))return this.isOdd()?E:y;if(this.isNegative())return e.isNegative()?this.neg().mul(e.neg()):this.neg().mul(e).neg();if(e.isNegative())return this.mul(e.neg()).neg();if(this.lt(v)&&e.lt(v))return c(this.toNumber()*e.toNumber(),this.unsigned);var t=this.high>>>16,r=this.high&65535,a=this.low>>>16,o=this.low&65535,s=e.high>>>16,u=e.high&65535,d=e.low>>>16,p=e.low&65535,m=0,h=0,g=0,_=0;return _+=o*p,g+=_>>>16,_&=65535,g+=a*p,h+=g>>>16,g&=65535,g+=o*d,h+=g>>>16,g&=65535,h+=r*p,m+=h>>>16,h&=65535,h+=a*d,m+=h>>>16,h&=65535,h+=o*u,m+=h>>>16,h&=65535,m+=t*p+r*d+a*u+o*s,m&=65535,l(g<<16|_,m<<16|h,this.unsigned)},D.mul=D.multiply,D.divide=function(e){if(i(e)||(e=f(e)),e.isZero())throw Error(`division by zero`);if(n)return!this.unsigned&&this.high===-2147483648&&e.low===-1&&e.high===-1?this:l((this.unsigned?n.div_u:n.div_s)(this.low,this.high,e.low,e.high),n.get_high(),this.unsigned);if(this.isZero())return this.unsigned?b:y;var t,r,a;if(this.unsigned){if(e.unsigned||(e=e.toUnsigned()),e.gt(this))return b;if(e.gt(this.shru(1)))return S;a=b}else{if(this.eq(E))return e.eq(x)||e.eq(C)?E:e.eq(E)?x:(t=this.shr(1).div(e).shl(1),t.eq(y)?e.isNegative()?x:C:(r=this.sub(e.mul(t)),a=t.add(r.div(e)),a));if(e.eq(E))return this.unsigned?b:y;if(this.isNegative())return e.isNegative()?this.neg().div(e.neg()):this.neg().div(e).neg();if(e.isNegative())return this.div(e.neg()).neg();a=y}for(r=this;r.gte(e);){t=Math.max(1,Math.floor(r.toNumber()/e.toNumber()));for(var o=Math.ceil(Math.log(t)/Math.LN2),s=o<=48?1:u(2,o-48),d=c(t),p=d.mul(e);p.isNegative()||p.gt(r);)t-=s,d=c(t,this.unsigned),p=d.mul(e);d.isZero()&&(d=x),a=a.add(d),r=r.sub(p)}return a},D.div=D.divide,D.modulo=function(e){return i(e)||(e=f(e)),n?l((this.unsigned?n.rem_u:n.rem_s)(this.low,this.high,e.low,e.high),n.get_high(),this.unsigned):this.sub(this.div(e).mul(e))},D.mod=D.modulo,D.rem=D.modulo,D.not=function(){return l(~this.low,~this.high,this.unsigned)},D.and=function(e){return i(e)||(e=f(e)),l(this.low&e.low,this.high&e.high,this.unsigned)},D.or=function(e){return i(e)||(e=f(e)),l(this.low|e.low,this.high|e.high,this.unsigned)},D.xor=function(e){return i(e)||(e=f(e)),l(this.low^e.low,this.high^e.high,this.unsigned)},D.shiftLeft=function(e){return i(e)&&(e=e.toInt()),(e&=63)==0?this:e<32?l(this.low<<e,this.high<<e|this.low>>>32-e,this.unsigned):l(0,this.low<<e-32,this.unsigned)},D.shl=D.shiftLeft,D.shiftRight=function(e){return i(e)&&(e=e.toInt()),(e&=63)==0?this:e<32?l(this.low>>>e|this.high<<32-e,this.high>>e,this.unsigned):l(this.high>>e-32,this.high>=0?0:-1,this.unsigned)},D.shr=D.shiftRight,D.shiftRightUnsigned=function(e){if(i(e)&&(e=e.toInt()),e&=63,e===0)return this;var t=this.high;if(e<32){var n=this.low;return l(n>>>e|t<<32-e,t>>>e,this.unsigned)}return l(e===32?t:t>>>e-32,0,this.unsigned)},D.shru=D.shiftRightUnsigned,D.shr_u=D.shiftRightUnsigned,D.toSigned=function(){return this.unsigned?l(this.low,this.high,!1):this},D.toUnsigned=function(){return this.unsigned?this:l(this.low,this.high,!0)},D.toBytes=function(e){return e?this.toBytesLE():this.toBytesBE()},D.toBytesLE=function(){var e=this.high,t=this.low;return[t&255,t>>>8&255,t>>>16&255,t>>>24,e&255,e>>>8&255,e>>>16&255,e>>>24]},D.toBytesBE=function(){var e=this.high,t=this.low;return[e>>>24,e>>>16&255,e>>>8&255,e&255,t>>>24,t>>>16&255,t>>>8&255,t&255]},r.fromBytes=function(e,t,n){return n?r.fromBytesLE(e,t):r.fromBytesBE(e,t)},r.fromBytesLE=function(e,t){return new r(e[0]|e[1]<<8|e[2]<<16|e[3]<<24,e[4]|e[5]<<8|e[6]<<16|e[7]<<24,t)},r.fromBytesBE=function(e,t){return new r(e[4]<<24|e[5]<<16|e[6]<<8|e[7],e[0]<<24|e[1]<<16|e[2]<<8|e[3],t)}}))());
/**
* @license
* Copyright 2021 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let ar=ir.default||ir;function or(e){return ar.fromString(e,!0,16)}let sr=or(`c3a5c85c97cb3127`),cr=or(`b492b66fbe98f273`),lr=or(`9ae16a3b2f90404f`);function ur(e){return e.xor(e.shru(47))}function dr(e,t,n){let r=e.slice(t,t+n);return ar.fromBytes(Array.from(r),!0,!0)}function V(e,t){return dr(e,t,8)}function fr(e,t){return dr(e,t,4)}function pr(e,t){return t===0?e:e.shru(t).or(e.shl(64-t))}function mr(e,t,n=or(`9ddfea08eb382d69`)){let r=e.xor(t).mul(n);r=r.xor(r.shru(47));let i=t.xor(r).mul(n);return i=i.xor(i.shru(47)),i=i.mul(n),i}function hr(e,t,n,r,i,a){i=i.add(e),a=pr(a.add(i).add(r),21);let o=i;return i=i.add(t),i=i.add(n),a=a.add(pr(i,44)),[i.add(r),a.add(o)]}function gr(e,t,n,r){return hr(V(e,t),V(e,t+8),V(e,t+16),V(e,t+24),n,r)}function _r(e,t=e.length){if(t>=8){let n=lr.add(t*2),r=V(e,0).add(lr),i=V(e,t-8);return mr(pr(i,37).mul(n).add(r),pr(r,25).add(i).mul(n),n)}if(t>=4){let n=lr.add(t*2);return mr(fr(e,0).shl(3).add(t),fr(e,t-4),n)}if(t>0){let n=e[0],r=e[t>>1],i=e[t-1],a=n+(r<<8),o=t+(i<<2);return ur(lr.mul(a).xor(sr.mul(o))).mul(lr)}return lr}function vr(e,t=e.length){let n=lr.add(t*2),r=V(e,0).mul(cr),i=V(e,8),a=V(e,t-8).mul(n),o=V(e,t-16).mul(lr);return mr(pr(r.add(i),43).add(pr(a,30)).add(o),r.add(pr(i.add(lr),18)).add(a),n)}function yr(e,t=e.length){let n=lr.add(t*2),r=V(e,0).mul(lr),i=V(e,8),a=V(e,t-8).mul(n),o=V(e,t-16).mul(lr),s=pr(r.add(i),43).add(pr(a,30)).add(o),c=mr(s,r.add(pr(i.add(lr),18)).add(a),n),l=V(e,16).mul(n),u=V(e,24),d=s.add(V(e,t-32)).mul(n),f=c.add(V(e,t-24)).mul(n);return mr(pr(l.add(u),43).add(pr(d,30)).add(f),l.add(pr(u.add(r),18)).add(d),n)}function br(e,t=e.length){let n=ar.fromNumber(81,!0);if(t<=32)return t<=16?_r(e,t):vr(e,t);if(t<=64)return yr(e,t);let r=n,i=n.mul(cr).add(113),a=ur(i.mul(lr).add(113)).mul(lr),o=[ar.UZERO,ar.UZERO],s=[ar.UZERO,ar.UZERO];r=r.mul(lr).add(V(e,0));let c=0,l=(t-1>>6)*64,u=l+(t-1&63)-63;do r=pr(r.add(i).add(o[0]).add(V(e,c+8)),37).mul(cr),i=pr(i.add(o[1]).add(V(e,c+48)),42).mul(cr),r=r.xor(s[1]),i=i.add(o[0]).add(V(e,c+40)),a=pr(a.add(s[0]),33).mul(cr),o=gr(e,c,o[1].mul(cr),r.add(s[0])),s=gr(e,c+32,a.add(s[1]),i.add(V(e,c+16))),[a,r]=[r,a],c+=64;while(c!==l);let d=cr.add(a.and(255).shl(1));return c=u,s[0]=s[0].add(t-1&63),o[0]=o[0].add(s[0]),s[0]=s[0].add(o[0]),r=pr(r.add(i).add(o[0]).add(V(e,c+8)),37).mul(d),i=pr(i.add(o[1]).add(V(e,c+48)),42).mul(d),r=r.xor(s[1].mul(9)),i=i.add(o[0].mul(9).add(V(e,c+40))),a=pr(a.add(s[0]),33).mul(d),o=gr(e,c,o[1].mul(d),r.add(s[0])),s=gr(e,c+32,a.add(s[1]),i.add(V(e,c+16))),[a,r]=[r,a],mr(mr(o[0],s[0],d).add(ur(i).mul(sr)).add(a),mr(o[1],s[1],d).add(r),d)}
/**
* @license
* Copyright 2017 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function xr(e,t){return t===`string`?Er(e):Cr([e],t)}function Sr(e,t){return e instanceof Float32Array&&t===`float32`||e instanceof Int32Array&&t===`int32`||e instanceof Uint8Array&&t===`bool`}function Cr(e,t){if(t===`string`)throw Error(`Cannot convert a string[] to a TypedArray`);if(Array.isArray(e)&&(e=kr(e)),z().getBool(`DEBUG`)&&T(e,t),Sr(e,t))return e;if(t==null||t===`float32`||t===`complex64`)return new Float32Array(e);if(t===`int32`)return new Int32Array(e);if(t===`bool`){let t=new Uint8Array(e.length);for(let n=0;n<t.length;++n)Math.round(e[n])!==0&&(t[n]=1);return t}throw Error(`Unknown data type ${t}`)}function wr(){return z().platform.now()}function Tr(e,t){return z().platform.fetch(e,t)}function Er(e,t=`utf-8`){return t||=`utf-8`,z().platform.encode(e,t)}function Dr(e,t=`utf-8`){return t||=`utf-8`,z().platform.decode(e,t)}function Or(e){return z().platform.isTypedArray==null?rr(e):z().platform.isTypedArray(e)}function kr(e,t=[],n=!1){if(t??=[],typeof e==`boolean`||typeof e==`number`||typeof e==`string`||oe(e)||e==null||Or(e)&&n)t.push(e);else if(Array.isArray(e)||Or(e))for(let r=0;r<e.length;++r)kr(e[r],t,n);else{let r=-1;for(let t of Object.keys(e))/^([1-9]+[0-9]*|0)$/.test(t)&&(r=Math.max(r,Number(t)));for(let i=0;i<=r;i++)kr(e[i],t,n)}return t}
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
var Ar=class{constructor(e,t){this.backendTimer=e,this.logger=t,t??(this.logger=new Mr)}profileKernel(e,t,n){let r,i=()=>{r=n()},a,o=wr();if(this.backendTimer.timerAvailable())a=this.backendTimer.time(i);else{i();for(let e of r)e.dataSync();a=Promise.resolve({kernelMs:wr()-o})}if(z().getBool(`CHECK_COMPUTATION_FOR_ERRORS`))for(let t=0;t<r.length;t++){let n=r[t];n.data().then(t=>{jr(t,n.dtype,e)})}return{kernelName:e,outputs:r,inputs:t,timeMs:a.then(e=>e.kernelMs),extraInfo:a.then(e=>e.getExtraProfileInfo==null?``:e.getExtraProfileInfo())}}logKernelProfile(e){let{kernelName:t,outputs:n,timeMs:r,inputs:i,extraInfo:a}=e;n.forEach(e=>{Promise.all([e.data(),r,a]).then(n=>{this.logger.logKernelProfile(t,e,n[0],n[1],i,n[2])})})}};function jr(e,t,n){if(t!==`float32`)return!1;for(let t=0;t<e.length;t++){let r=e[t];if(isNaN(r)||!isFinite(r))return console.warn(`Found ${r} in the result of '${n}'`),!0}return!1}var Mr=class{logKernelProfile(e,t,n,r,i,a){let o=typeof r==`number`?b(`${r}ms`,9):r.error,s=b(e,25),c=t.rank,l=t.size,u=b(t.shape.toString(),14),d=``;for(let e in i){let n=i[e];if(n!=null){let r=n.shape||t.shape,i=r.length;d+=`${e}: ${i}D ${i>0?r:``} `}}console.log(`%c${s}\t%c${o}\t%c${c}D ${u}\t%c${l}\t%c${d}\t%c${a}`,`font-weight:bold`,`color:red`,`color:blue`,`color: orange`,`color: green`,`color: steelblue`)}};
/**
* @license
* Copyright 2017 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Nr(e,t,n){let r={},i={};for(let e=0;e<t.length;e++)r[t[e].id]=!0;for(let n=0;n<e.length;n++){let a=e[n],o=a.inputs;for(let e in o){let n=o[e],s=!1;for(let e=0;e<t.length;e++)if(r[n.id]){a.outputs.forEach(e=>r[e.id]=!0),s=!0,i[a.id]=!0;break}if(s)break}}let a={};a[n.id]=!0;let o={};for(let t=e.length-1;t>=0;t--){let n=e[t],r=n.inputs;for(let e=0;e<n.outputs.length;e++)if(a[n.outputs[e].id]){for(let e in r)a[r[e].id]=!0,o[n.id]=!0;break}}let s=[];for(let t=0;t<e.length;t++){let n=e[t];if(i[n.id]&&o[n.id]){let e={};for(let t in n.inputs){let i=n.inputs[t];r[i.id]&&(e[t]=i)}let t=Object.assign({},n);t.inputs=e,t.outputs=n.outputs,s.push(t)}}return s}function Pr(e,t,n,r){for(let i=t.length-1;i>=0;i--){let a=t[i],o=[];if(a.outputs.forEach(t=>{let n=e[t.id];n==null?o.push(null):o.push(n)}),a.gradient==null)throw Error(`Cannot compute gradient: gradient function not found for ${a.kernelName}.`);let s=a.gradient(o);for(let t in a.inputs){if(!(t in s))throw Error(`Cannot backprop through input ${t}. Available gradients found: ${Object.keys(s)}.`);let i=n(()=>s[t]());if(i.dtype!==`float32`)throw Error(`Error in gradient for op ${a.kernelName}. The gradient of input ${t} must have 'float32' dtype, but has '${i.dtype}'`);let o=a.inputs[t];if(!v(i.shape,o.shape))throw Error(`Error in gradient for op ${a.kernelName}. The gradient of input '${t}' has shape '${i.shape}', which does not match the shape of the input '${o.shape}'`);if(e[o.id]==null)e[o.id]=i;else{let t=e[o.id];e[o.id]=r(t,i),t.dispose()}}}}
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Fr(e,t,n,r){let i=P(t),a=Ir(e,t,n,i),o=t.length,s=zr(e,t,n,i,a),c=[`Tensor`];return r&&(c.push(`  dtype: ${n}`),c.push(`  rank: ${o}`),c.push(`  shape: [${t}]`),c.push(`  values:`)),c.push(s.map(e=>`    `+e).join(`
`)),c.join(`
`)}function Ir(e,t,n,r){let i=_(t),a=r[r.length-1],o=Array(a).fill(0),s=t.length,c=n===`complex64`?Br(e):e;if(s>1)for(let e=0;e<i/a;e++){let t=e*a;for(let e=0;e<a;e++)o[e]=Math.max(o[e],Lr(c[t+e],0,n).length)}return o}function Lr(e,t,n){let r;return r=Array.isArray(e)?`${parseFloat(e[0].toFixed(7))} + ${parseFloat(e[1].toFixed(7))}j`:A(e)?`'${e}'`:n===`bool`?Rr(e):parseFloat(e.toFixed(7)).toString(),b(r,t)}function Rr(e){return e===0?`false`:`true`}function zr(e,t,n,r,i,a=!0){let o=n===`complex64`?2:1,s=t[0],c=t.length;if(c===0)return n===`complex64`?[Lr(Br(e)[0],0,n)]:n===`bool`?[Rr(e[0])]:[e[0].toString()];if(c===1){if(s>20){let t=3*o,r=Array.from(e.slice(0,t)),a=Array.from(e.slice((s-3)*o,s*o));return n===`complex64`&&(r=Br(r),a=Br(a)),[`[`+r.map((e,t)=>Lr(e,i[t],n)).join(`, `)+`, ..., `+a.map((e,t)=>Lr(e,i[s-3+t],n)).join(`, `)+`]`]}return[`[`+(n===`complex64`?Br(e):Array.from(e)).map((e,t)=>Lr(e,i[t],n)).join(`, `)+`]`]}let l=t.slice(1),u=r.slice(1),d=r[0]*o,f=[];if(s>20){for(let t=0;t<3;t++){let r=t*d,a=r+d;f.push(...zr(e.slice(r,a),l,n,u,i,!1))}f.push(`...`);for(let t=s-3;t<s;t++){let r=t*d,a=r+d;f.push(...zr(e.slice(r,a),l,n,u,i,t===s-1))}}else for(let t=0;t<s;t++){let r=t*d,a=r+d;f.push(...zr(e.slice(r,a),l,n,u,i,t===s-1))}let p=c===2?`,`:``;f[0]=`[`+(s>0?f[0]+p:``);for(let e=1;e<f.length-1;e++)f[e]=` `+f[e]+p;let m=`,
`;for(let e=2;e<c;e++)m+=`
`;return f[f.length-1]=` `+f[f.length-1]+`]`+(a?``:m),f}function Br(e){let t=[];for(let n=0;n<e.length;n+=2)t.push([e[n],e[n+1]]);return t}
/**
* @license
* Copyright 2017 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
var Vr=class{constructor(e,t,n){if(this.dtype=t,this.shape=e.slice(),this.size=_(e),n!=null){let e=n.length;m(e===this.size,()=>`Length of values '${e}' does not match the size inferred by the shape '${this.size}'.`)}if(t===`complex64`)throw Error(`complex64 dtype TensorBuffers are not supported. Please create a TensorBuffer for the real and imaginary parts separately and call tf.complex(real, imag).`);this.values=n||w(t,this.size),this.strides=P(e)}set(e,...t){t.length===0&&(t=[0]),m(t.length===this.rank,()=>`The number of provided coordinates (${t.length}) must match the rank (${this.rank})`);let n=this.locToIndex(t);this.values[n]=e}get(...e){e.length===0&&(e=[0]);let t=0;for(let n of e){if(n<0||n>=this.shape[t]){let t=`Requested out of range element at ${e}.   Buffer shape=${this.shape}`;throw Error(t)}t++}let n=e[e.length-1];for(let t=0;t<e.length-1;++t)n+=this.strides[t]*e[t];return this.values[n]}locToIndex(e){if(this.rank===0)return 0;if(this.rank===1)return e[0];let t=e[e.length-1];for(let n=0;n<e.length-1;++n)t+=this.strides[n]*e[n];return t}indexToLoc(e){if(this.rank===0)return[];if(this.rank===1)return[e];let t=Array(this.shape.length);for(let n=0;n<t.length-1;++n)t[n]=Math.floor(e/this.strides[n]),e-=t[n]*this.strides[n];return t[t.length-1]=e,t}get rank(){return this.shape.length}toTensor(){return Hr().makeTensor(this.values,this.shape,this.dtype)}};let Hr=null,Ur=null;function Wr(e){Hr=e}function Gr(e){Ur=e}var Kr=class{constructor(e,t,n,r){this.kept=!1,this.isDisposedInternal=!1,this.shape=e.slice(),this.dtype=t||`float32`,this.size=_(e),this.strides=P(e),this.dataId=n,this.id=r,this.rankType=this.rank<5?this.rank.toString():`higher`}get rank(){return this.shape.length}async buffer(){let e=await this.data();return Ur.buffer(this.shape,this.dtype,e)}bufferSync(){return Ur.buffer(this.shape,this.dtype,this.dataSync())}async array(){let e=await this.data();return F(this.shape,e,this.dtype===`complex64`)}arraySync(){return F(this.shape,this.dataSync(),this.dtype===`complex64`)}async data(){this.throwIfDisposed();let e=Hr().read(this.dataId);if(this.dtype===`string`){let t=await e;try{return t.map(e=>Dr(e))}catch{throw Error(`Failed to decode the string bytes into utf-8. To get the original bytes, call tensor.bytes().`)}}return e}dataToGPU(e){return this.throwIfDisposed(),Hr().readToGPU(this.dataId,e)}dataSync(){this.throwIfDisposed();let e=Hr().readSync(this.dataId);if(this.dtype===`string`)try{return e.map(e=>Dr(e))}catch{throw Error(`Failed to decode the string bytes into utf-8. To get the original bytes, call tensor.bytes().`)}return e}async bytes(){this.throwIfDisposed();let e=await Hr().read(this.dataId);return this.dtype===`string`?e:new Uint8Array(e.buffer)}dispose(){this.isDisposed||(this.kerasMask&&this.kerasMask.dispose(),Hr().disposeTensor(this),this.isDisposedInternal=!0)}get isDisposed(){return this.isDisposedInternal}throwIfDisposed(){if(this.isDisposed)throw Error(`Tensor is disposed.`)}print(e=!1){return Ur.print(this,e)}clone(){return this.throwIfDisposed(),Ur.clone(this)}toString(e=!1){return Fr(this.dataSync(),this.shape,this.dtype,e)}cast(e){return this.throwIfDisposed(),Ur.cast(this,e)}variable(e=!0,t,n){return this.throwIfDisposed(),Hr().makeVariable(this,e,t,n)}};Object.defineProperty(Kr,Symbol.hasInstance,{value:e=>!!e&&e.data!=null&&e.dataSync!=null&&e.throwIfDisposed!=null});function qr(){return _e(`Tensor`,()=>Kr)}qr();var Jr=class extends Kr{constructor(e,t,n,r){super(e.shape,e.dtype,e.dataId,r),this.trainable=t,this.name=n}assign(e){if(e.dtype!==this.dtype)throw Error(`dtype of the new value (${e.dtype}) and previous value (${this.dtype}) must match`);if(!v(e.shape,this.shape))throw Error(`shape of the new value (${e.shape}) and previous value (${this.shape}) must match`);Hr().disposeTensor(this),this.dataId=e.dataId,Hr().incRef(this,null)}dispose(){Hr().disposeVariable(this),this.isDisposedInternal=!0}};Object.defineProperty(Jr,Symbol.hasInstance,{value:e=>e instanceof Kr&&e.assign!=null&&e.assign instanceof Function});
/**
* @license
* Copyright 2017 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
var Yr;(function(e){e.R0=`R0`,e.R1=`R1`,e.R2=`R2`,e.R3=`R3`,e.R4=`R4`,e.R5=`R5`,e.R6=`R6`})(Yr||={});var Xr;(function(e){e.float32=`float32`,e.int32=`int32`,e.bool=`int32`,e.complex64=`complex64`})(Xr||={});var Zr;(function(e){e.float32=`float32`,e.int32=`int32`,e.bool=`bool`,e.complex64=`complex64`})(Zr||={});var Qr;(function(e){e.float32=`float32`,e.int32=`float32`,e.bool=`float32`,e.complex64=`complex64`})(Qr||={});var $r;(function(e){e.float32=`complex64`,e.int32=`complex64`,e.bool=`complex64`,e.complex64=`complex64`})($r||={});let ei={float32:Qr,int32:Xr,bool:Zr,complex64:$r};function ti(e,t){if(e===`string`||t===`string`){if(e===`string`&&t===`string`)return`string`;throw Error(`Can not upcast ${e} with ${t}`)}return ei[e][t]}function ni(e){return typeof e==`object`&&!!e&&`texture`in e&&e.texture instanceof WebGLTexture}function ri(e){return typeof GPUBuffer<`u`&&typeof e==`object`&&!!e&&`buffer`in e&&e.buffer instanceof GPUBuffer}
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function ii(e,t){if(e.dtype===t.dtype)return[e,t];let n=ti(e.dtype,t.dtype);return[e.cast(n),t.cast(n)]}function ai(e){let t=[];return oi(e,t,new Set),t}function oi(e,t,n){if(e==null)return;if(e instanceof Kr){t.push(e);return}if(!si(e))return;let r=e;for(let e in r){let i=r[e];n.has(i)||(n.add(i),oi(i,t,n))}}function si(e){return Array.isArray(e)||typeof e==`object`}
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function ci(e){return e.kernelName!=null}var li=class{constructor(){this.registeredVariables={},this.nextTapeNodeId=0,this.numBytes=0,this.numTensors=0,this.numStringTensors=0,this.numDataBuffers=0,this.gradientDepth=0,this.kernelDepth=0,this.scopeStack=[],this.numDataMovesStack=[],this.nextScopeId=0,this.tensorInfo=new WeakMap,this.profiling=!1,this.activeProfile={newBytes:0,newTensors:0,peakBytes:0,kernels:[],result:null,get kernelNames(){return Array.from(new Set(this.kernels.map(e=>e.name)))}}}dispose(){for(let e in this.registeredVariables)this.registeredVariables[e].dispose()}},ui=class e{constructor(e){this.ENV=e,this.registry={},this.registryFactory={},this.pendingBackendInitId=0,this.state=new li}async ready(){if(this.pendingBackendInit!=null)return this.pendingBackendInit.then(()=>{});if(this.backendInstance!=null)return;let e=this.getSortedBackends();for(let t=0;t<e.length;t++){let n=e[t];if(await this.initializeBackend(n).success){await this.setBackend(n);return}}throw Error(`Could not initialize any backends, all backend initializations failed.`)}get backend(){if(this.pendingBackendInit!=null)throw Error(`Backend '${this.backendName}' has not yet been initialized. Make sure to await tf.ready() or await tf.setBackend() before calling other methods`);if(this.backendInstance==null){let{name:e,asyncInit:t}=this.initializeBackendsAndReturnBest();if(t)throw Error(`The highest priority backend '${e}' has not yet been initialized. Make sure to await tf.ready() or await tf.setBackend() before calling other methods`);this.setBackend(e)}return this.backendInstance}backendNames(){return Object.keys(this.registryFactory)}findBackend(e){if(!(e in this.registry)){if(e in this.registryFactory){let{asyncInit:t}=this.initializeBackend(e);if(t)return null}else return null}return this.registry[e]}findBackendFactory(e){return e in this.registryFactory?this.registryFactory[e].factory:null}registerBackend(e,t,n=1){return e in this.registryFactory?(Yn(`${e} backend was already registered. Reusing existing backend factory.`),!1):(this.registryFactory[e]={factory:t,priority:n},!0)}async setBackend(e){if(this.registryFactory[e]==null)throw Error(`Backend name '${e}' not found in registry`);if(this.backendName=e,this.registry[e]==null){this.backendInstance=null;let{success:t,asyncInit:n}=this.initializeBackend(e);if(!(n?await t:t))return!1}return this.backendInstance=this.registry[e],this.setupRegisteredKernels(),this.profiler=new Ar(this.backendInstance),!0}setupRegisteredKernels(){er(this.backendName).forEach(e=>{e.setupFunc!=null&&e.setupFunc(this.backendInstance)})}disposeRegisteredKernels(e){er(e).forEach(t=>{t.disposeFunc!=null&&t.disposeFunc(this.registry[e])})}initializeBackend(e){let t=this.registryFactory[e];if(t==null)throw Error(`Cannot initialize backend ${e}, no registration found.`);try{let n=t.factory();if(n&&!(n instanceof u)&&typeof n.then==`function`){let t=++this.pendingBackendInitId,r=n.then(n=>t<this.pendingBackendInitId?!1:(this.registry[e]=n,this.pendingBackendInit=null,!0)).catch(n=>t<this.pendingBackendInitId?!1:(this.pendingBackendInit=null,Yn(`Initialization of backend ${e} failed`),Yn(n.stack||n.message),!1));return this.pendingBackendInit=r,{success:r,asyncInit:!0}}return this.registry[e]=n,{success:!0,asyncInit:!1}}catch(t){return Yn(`Initialization of backend ${e} failed`),Yn(t.stack||t.message),{success:!1,asyncInit:!1}}}removeBackend(e){if(!(e in this.registryFactory))throw Error(`${e} backend not found in registry`);this.backendName===e&&this.pendingBackendInit!=null&&this.pendingBackendInitId++,e in this.registry&&(this.disposeRegisteredKernels(e),this.registry[e].dispose(),delete this.registry[e]),delete this.registryFactory[e],this.backendName===e&&(this.pendingBackendInit=null,this.backendName=null,this.backendInstance=null)}getSortedBackends(){if(Object.keys(this.registryFactory).length===0)throw Error(`No backend found in registry.`);return Object.keys(this.registryFactory).sort((e,t)=>this.registryFactory[t].priority-this.registryFactory[e].priority)}initializeBackendsAndReturnBest(){let e=this.getSortedBackends();for(let t=0;t<e.length;t++){let n=e[t],{success:r,asyncInit:i}=this.initializeBackend(n);if(i||r)return{name:n,asyncInit:i}}throw Error(`Could not initialize any backends, all backend initializations failed.`)}moveData(e,t){let n=this.state.tensorInfo.get(t),r=n.backend,i=this.readSync(t),a=r.refCount(t);r.disposeData(t,!0),n.backend=e,e.move(t,i,n.shape,n.dtype,a),this.shouldCheckForMemLeaks()&&this.state.numDataMovesStack[this.state.numDataMovesStack.length-1]++}tidy(e,t){let n=null;if(t==null){if(typeof e!=`function`)throw Error(`Please provide a function to tidy()`);t=e}else{if(typeof e!=`string`&&!(e instanceof String))throw Error(`When calling with two arguments, the first argument to tidy() must be a string`);if(typeof t!=`function`)throw Error(`When calling with two arguments, the 2nd argument to tidy() must be a function`);n=e}let r;return this.scopedRun(()=>this.startScope(n),()=>this.endScope(r),()=>(r=t(),r instanceof Promise&&console.error(`Cannot return a Promise inside of tidy.`),r))}scopedRun(e,t,n){e();try{let e=n();return t(),e}catch(e){throw t(),e}}nextTensorId(){return e.nextTensorId++}nextVariableId(){return e.nextVariableId++}clone(e){let t=H.runKernel(_t,{x:e}),n={x:e};return this.addTapeNode(this.state.activeScope.name,n,[t],e=>({x:()=>{let t={x:e};return H.runKernel(Le,t,{dtype:`float32`})}}),[],{}),t}runKernel(e,t,n){if(this.backendName??this.backend,Qn(e,this.backendName)==null)throw Error(`Kernel '${e}' not registered for backend '${this.backendName}'`);return this.runKernelFunc({kernelName:e,inputs:t,attrs:n})}shouldCheckForMemLeaks(){return this.ENV.getBool(`IS_TEST`)}checkKernelForMemLeak(e,t,n){let r=this.backend.numDataIds(),i=0;n.forEach(e=>{i+=e.dtype===`complex64`?3:1});let a=this.state.numDataMovesStack[this.state.numDataMovesStack.length-1],o=r-t-i-a;if(o>0)throw Error(`Backend '${this.backendName}' has an internal memory leak (${o} data ids) after running '${e}'`)}runKernelFunc(e){let t,n=[],r=this.isTapeOn(),i=this.state.numBytes,a=this.state.numTensors;this.shouldCheckForMemLeaks()&&this.state.numDataMovesStack.push(0);let o;this.backendName??this.backend;let s,c=ci(e)?e.kernelName:this.state.activeScope==null?``:this.state.activeScope.name;if(ci(e)){let{kernelName:t,inputs:i,attrs:a}=e;this.backendName??this.backend;let c=Qn(t,this.backendName);m(c!=null,()=>`Cannot find registered kernel '${t}' for backend '${this.backendName}'`),o=()=>{let e=this.backend.numDataIds();s=c.kernelFunc({inputs:i,attrs:a,backend:this.backend});let o=Array.isArray(s)?s:[s];this.shouldCheckForMemLeaks()&&this.checkKernelForMemLeak(t,e,o);let l=o.map(e=>e.rank==null?this.makeTensorFromTensorInfo(e):e);if(r){let e=this.getTensorsForGradient(t,i,l);n=this.saveTensorsForBackwardMode(e)}return l}}else{let{forwardFunc:t}=e,i=e=>{r&&(n=e.map(e=>this.keep(this.clone(e))))};o=()=>{let e=this.backend.numDataIds();s=this.tidy(()=>t(this.backend,i));let n=Array.isArray(s)?s:[s];return this.shouldCheckForMemLeaks()&&this.checkKernelForMemLeak(c,e,n),n}}let{inputs:l,attrs:u}=e,d=ci(e)?null:e.backwardsFunc,f;return this.scopedRun(()=>this.state.kernelDepth++,()=>this.state.kernelDepth--,()=>{!this.ENV.getBool(`DEBUG`)&&!this.state.profiling?t=o():(f=this.profiler.profileKernel(c,l,()=>o()),this.ENV.getBool(`DEBUG`)&&this.profiler.logKernelProfile(f),t=f.outputs)}),r&&this.addTapeNode(c,l,t,d,n,u),this.state.profiling&&this.state.activeProfile.kernels.push({name:c,bytesAdded:this.state.numBytes-i,totalBytesSnapshot:this.state.numBytes,tensorsAdded:this.state.numTensors-a,totalTensorsSnapshot:this.state.numTensors,inputShapes:Object.keys(l).map(e=>l[e]==null?null:l[e].shape),outputShapes:t.map(e=>e.shape),kernelTimeMs:f.timeMs,extraInfo:f.extraInfo}),Array.isArray(s)?t:t[0]}saveTensorsForBackwardMode(e){return e.map(e=>this.keep(this.clone(e)))}getTensorsForGradient(e,t,n){let r=$n(e);if(r!=null){let e=r.inputsToSave||[],i=r.outputsToSave||[],a;r.saveAllInputs?(m(Array.isArray(t),()=>`saveAllInputs is true, expected inputs to be an array.`),a=Object.keys(t).map(e=>t[e])):a=e.map(e=>t[e]);let o=n.filter((e,t)=>i[t]);return a.concat(o)}return[]}makeTensor(e,t,n,r){if(e==null)throw Error(`Values passed to engine.makeTensor() are null`);n||=`float32`,r||=this.backend;let i=e;n===`string`&&A(e[0])&&(i=e.map(e=>Er(e)));let a=r.write(i,t,n),o=new Kr(t,n,a,this.nextTensorId());if(this.trackTensor(o,r),n===`string`){let e=this.state.tensorInfo.get(a),t=k(i);this.state.numBytes+=t-e.bytes,e.bytes=t}return o}makeTensorFromDataId(e,t,n,r){n||=`float32`;let i={dataId:e,shape:t,dtype:n};return this.makeTensorFromTensorInfo(i,r)}makeTensorFromTensorInfo(e,t){let{dataId:n,shape:r,dtype:i}=e,a=new Kr(r,i,n,this.nextTensorId());return this.trackTensor(a,t),a}makeVariable(e,t=!0,n,r){n||=this.nextVariableId().toString(),r!=null&&r!==e.dtype&&(e=e.cast(r));let i=new Jr(e,t,n,this.nextTensorId());if(this.state.registeredVariables[i.name]!=null)throw Error(`Variable with name ${i.name} was already registered`);return this.state.registeredVariables[i.name]=i,this.incRef(i,this.backend),i}trackTensor(e,t){this.state.numTensors++,e.dtype===`string`&&this.state.numStringTensors++;let n=0;e.dtype!==`complex64`&&e.dtype!==`string`&&(n=e.size*O(e.dtype)),this.state.numBytes+=n,this.state.tensorInfo.has(e.dataId)||(this.state.numDataBuffers++,this.state.tensorInfo.set(e.dataId,{backend:t||this.backend,dtype:e.dtype,shape:e.shape,bytes:n})),e instanceof Jr||this.track(e)}incRef(e,t){this.trackTensor(e,t),this.backend.incRef(e.dataId)}removeDataId(e,t){this.state.tensorInfo.has(e)&&this.state.tensorInfo.get(e).backend===t&&(this.state.tensorInfo.delete(e),this.state.numDataBuffers--)}disposeTensor(e){if(!this.state.tensorInfo.has(e.dataId))return;let t=this.state.tensorInfo.get(e.dataId);if(this.state.numTensors--,e.dtype===`string`&&(this.state.numStringTensors--,this.state.numBytes-=t.bytes),e.dtype!==`complex64`&&e.dtype!==`string`){let t=e.size*O(e.dtype);this.state.numBytes-=t}t.backend.disposeData(e.dataId)&&this.removeDataId(e.dataId,t.backend)}disposeVariables(){for(let e in this.state.registeredVariables){let t=this.state.registeredVariables[e];this.disposeVariable(t)}}disposeVariable(e){this.disposeTensor(e),this.state.registeredVariables[e.name]!=null&&delete this.state.registeredVariables[e.name]}memory(){let e=this.backend.memory();return e.numTensors=this.state.numTensors,e.numDataBuffers=this.state.numDataBuffers,e.numBytes=this.state.numBytes,this.state.numStringTensors>0&&(e.unreliable=!0,e.reasons??=[],e.reasons.push(`Memory usage by string tensors is approximate (2 bytes per character)`)),e}async profile(e){this.state.profiling=!0;let t=this.state.numBytes,n=this.state.numTensors;this.state.activeProfile.kernels=[],this.state.activeProfile.result=await e(),this.state.profiling=!1,this.state.activeProfile.peakBytes=Math.max(...this.state.activeProfile.kernels.map(e=>e.totalBytesSnapshot)),this.state.activeProfile.newBytes=this.state.numBytes-t,this.state.activeProfile.newTensors=this.state.numTensors-n;for(let e of this.state.activeProfile.kernels)e.kernelTimeMs=await e.kernelTimeMs,e.extraInfo=await e.extraInfo;return this.state.activeProfile}isTapeOn(){return this.state.gradientDepth>0&&this.state.kernelDepth===0}addTapeNode(e,t,n,r,i,a){let o={id:this.state.nextTapeNodeId++,kernelName:e,inputs:t,outputs:n,saved:i},s=$n(e);s!=null&&(r=s.gradFunc),r!=null&&(o.gradient=e=>(e=e.map((e,t)=>{if(e==null){let e=n[t],r=I(e.size,e.dtype);return this.makeTensor(r,e.shape,e.dtype)}return e}),r(e.length>1?e:e[0],i,a))),this.state.activeTape.push(o)}keep(e){return e.kept=!0,e}startTape(){this.state.gradientDepth===0&&(this.state.activeTape=[]),this.state.gradientDepth++}endTape(){this.state.gradientDepth--}startScope(e){let t={track:[],name:`unnamed scope`,id:this.state.nextScopeId++};e&&(t.name=e),this.state.scopeStack.push(t),this.state.activeScope=t}endScope(e){let t=ai(e),n=new Set(t.map(e=>e.id));for(let e=0;e<this.state.activeScope.track.length;e++){let t=this.state.activeScope.track[e];!t.kept&&!n.has(t.id)&&t.dispose()}let r=this.state.scopeStack.pop();this.state.activeScope=this.state.scopeStack.length===0?null:this.state.scopeStack[this.state.scopeStack.length-1],t.forEach(e=>{!e.kept&&e.scopeId===r.id&&this.track(e)})}gradients(e,t,n,r=!1){if(m(t.length>0,()=>`gradients() received an empty list of xs.`),n!=null&&n.dtype!==`float32`)throw Error(`dy must have 'float32' dtype, but has '${n.dtype}'`);let i=this.scopedRun(()=>this.startTape(),()=>this.endTape(),()=>this.tidy(`forward`,e));m(i instanceof Kr,()=>`The result y returned by f() must be a tensor.`);let a=Nr(this.state.activeTape,t,i);if(!r&&a.length===0&&t.length>0)throw Error(`Cannot compute gradient of y=f(x) with respect to x. Make sure that the f you passed encloses all operations that lead from x to y.`);return this.tidy(`backward`,()=>{let e={};e[i.id]=n??di(i.shape),Pr(e,a,e=>this.tidy(e),pi);let r=t.map(t=>e[t.id]);return this.state.gradientDepth===0&&(this.state.activeTape.forEach(e=>{for(let t of e.saved)t.dispose()}),this.state.activeTape=null),{value:i,grads:r}})}customGrad(e){return m(ee(e),()=>`The f passed in customGrad(f) must be a function.`),(...t)=>{m(t.every(e=>e instanceof Kr),()=>`The args passed in customGrad(f)(x1, x2,...) must all be tensors`);let n,r={};return t.forEach((e,t)=>{r[t]=e}),this.runKernelFunc({forwardFunc:(r,i)=>(n=e(...t,i),m(n.value instanceof Kr,()=>"The function f passed in customGrad(f) must return an object where `obj.value` is a tensor"),m(ee(n.gradFunc),()=>"The function f passed in customGrad(f) must return an object where `obj.gradFunc` is a function."),n.value),backwardsFunc:(e,r)=>{let i=n.gradFunc(e,r),a=Array.isArray(i)?i:[i];m(a.length===t.length,()=>"The function f passed in customGrad(f) must return an object where `obj.gradFunc` is a function that returns the same number of tensors as inputs passed to f(...)."),m(a.every(e=>e instanceof Kr),()=>"The function f passed in customGrad(f) must return an object where `obj.gradFunc` is a function that returns a list of only tensors.");let o={};return a.forEach((e,t)=>{o[t]=()=>e}),o},inputs:r})}}readSync(e){return this.state.tensorInfo.get(e).backend.readSync(e)}read(e){return this.state.tensorInfo.get(e).backend.read(e)}readToGPU(e,t){return this.state.tensorInfo.get(e).backend.readToGPU(e,t)}async time(e){let t=wr(),n=await this.backend.time(e);return n.wallMs=wr()-t,n}track(e){return this.state.activeScope!=null&&(e.scopeId=this.state.activeScope.id,this.state.activeScope.track.push(e)),e}get registeredVariables(){return this.state.registeredVariables}reset(){this.pendingBackendInitId++,this.state.dispose(),this.ENV.reset(),this.state=new li;for(let e in this.registry)this.disposeRegisteredKernels(e),this.registry[e].dispose(),delete this.registry[e];this.backendName=null,this.backendInstance=null,this.pendingBackendInit=null}};ui.nextTensorId=0,ui.nextVariableId=0;function di(e){let t=re(_(e),`float32`);return H.makeTensor(t,e,`float32`)}function fi(){let e=he();return e._tfengine??=new ui(new ce(e)),pe(e._tfengine.ENV),Wr(()=>e._tfengine),e._tfengine}let H=fi();function pi(e,t){let n={a:e,b:t};return H.runKernel(`Add`,n)}
/**
* @license
* Copyright 2017 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function mi(){return typeof window<`u`&&window.document!=null||typeof WorkerGlobalScope<`u`}
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let hi=z();hi.registerFlag(`DEBUG`,()=>!1,e=>{e&&console.warn(`Debugging mode is ON. The output of every math call will be downloaded to CPU and checked for NaNs. This significantly impacts performance.`)}),hi.registerFlag(`IS_BROWSER`,()=>mi()),hi.registerFlag(`IS_NODE`,()=>typeof process<`u`&&process.versions!==void 0&&process.versions.node!==void 0),hi.registerFlag(`IS_CHROME`,()=>typeof navigator<`u`&&navigator!=null&&navigator.userAgent!=null&&/Chrome/.test(navigator.userAgent)&&/Google Inc/.test(navigator.vendor)),hi.registerFlag(`IS_SAFARI`,()=>typeof navigator<`u`&&navigator!=null&&navigator.userAgent!=null&&/Safari/.test(navigator.userAgent)&&/Apple/.test(navigator.vendor)),hi.registerFlag(`PROD`,()=>!1),hi.registerFlag(`TENSORLIKE_CHECK_SHAPE_CONSISTENCY`,()=>hi.getBool(`DEBUG`)),hi.registerFlag(`DEPRECATION_WARNINGS_ENABLED`,()=>!0),hi.registerFlag(`IS_TEST`,()=>!1),hi.registerFlag(`CHECK_COMPUTATION_FOR_ERRORS`,()=>hi.getBool(`DEBUG`)),hi.registerFlag(`WRAP_TO_IMAGEBITMAP`,()=>!1),hi.registerFlag(`CANVAS2D_WILL_READ_FREQUENTLY_FOR_GPU`,()=>!1),hi.registerFlag(`USE_SETTIMEOUTCUSTOM`,()=>!1);
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function gi(e,t){let n=e;if(Or(e))return t===`string`?[]:[e.length];if(ni(e)){let t=e.channels||`RGBA`;return[e.height,e.width*t.length]}if(ri(e))return[e.buffer.size/(t==null?4:O(t))];if(!Array.isArray(e))return[];let r=[];for(;Array.isArray(n)||Or(n)&&t!==`string`;)r.push(n.length),n=n[0];return Array.isArray(e)&&z().getBool(`TENSORLIKE_CHECK_SHAPE_CONSISTENCY`)&&_i(e,r,[]),r}function _i(e,t,n){if(n||=[],!Array.isArray(e)&&!Or(e)){m(t.length===0,()=>`Element arr[${n.join(`][`)}] is a primitive, but should be an array/TypedArray of ${t[0]} elements`);return}m(t.length>0,()=>`Element arr[${n.join(`][`)}] should be a primitive, but is an array of ${e.length} elements`),m(e.length===t[0],()=>`Element arr[${n.join(`][`)}] should have ${t[0]} elements, but has ${e.length} elements`);let r=t.slice(1);for(let t=0;t<e.length;++t)_i(e[t],r,n.concat(t))}function vi(e,t,n,r){if(e!==`string_or_numeric`){if(e==null)throw Error(`Expected dtype cannot be null.`);if(e!==`numeric`&&e!==t||e===`numeric`&&t===`string`)throw Error(`Argument '${n}' passed to '${r}' must be ${e} tensor, but got ${t} tensor`)}}function U(e,t,n,r=`numeric`){if(e instanceof qr())return vi(r,e.dtype,t,n),e;let i=N(e);if(i!==`string`&&[`bool`,`int32`,`float32`].indexOf(r)>=0&&(i=r),vi(r,i,t,n),e==null||!Or(e)&&!Array.isArray(e)&&typeof e!=`number`&&typeof e!=`boolean`&&typeof e!=`string`){let r=e==null?`null`:e.constructor.name;throw Error(`Argument '${t}' passed to '${n}' must be a Tensor or TensorLike, but got '${r}'`)}let a=gi(e,i);!Or(e)&&!Array.isArray(e)&&(e=[e]);let o=i===`string`?kr(e,[],!0):Cr(e,i);return H.makeTensor(o,a,i)}function yi(e,t,n,r=`numeric`){if(!Array.isArray(e))throw Error(`Argument ${t} passed to ${n} must be a \`Tensor[]\` or \`TensorLike[]\``);return e.map((e,i)=>U(e,`${t}[${i}]`,n,r))}
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function bi(e){let t=Object.keys(e);if(t.length!==1)throw Error(`Please provide an object with a single key (operation name) mapping to a function. Got an object with ${t.length} keys.`);let n=t[0],r=e[n];n.endsWith(`_`)&&(n=n.substring(0,n.length-1)),n+=`__op`;let i=(...e)=>{H.startScope(n);try{let t=r(...e);return oe(t)&&console.error(`Cannot return a Promise inside of tidy.`),H.endScope(t),t}catch(e){throw H.endScope(null),e}};return Object.defineProperty(i,"name",{value:n,configurable:!0}),i}
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function xi(e,t,n,r){if(r==null)r=N(e);else if(r===`complex64`)throw Error(`Cannot construct a complex64 tensor directly. Please use tf.complex(real, imag).`);if(ri(e)||ni(e)){if(r!==`float32`&&r!==`int32`)throw Error(`Creating tensor from GPU data only supports 'float32'|'int32' dtype, while the dtype is ${r}.`);return H.backend.createTensorFromGPUData(e,t||n,r)}if(!Or(e)&&!Array.isArray(e)&&typeof e!=`number`&&typeof e!=`boolean`&&typeof e!=`string`)throw Error(`values passed to tensor(values) must be a number/boolean/string or an array of numbers/booleans/strings, or a TypedArray`);if(t!=null){L(t);let e=_(t),r=_(n);m(e===r,()=>`Based on the provided shape, [${t}], the tensor should have ${e} values but has ${r}`);for(let e=0;e<n.length;++e){let r=n[e],i=e!==n.length-1||r!==_(t.slice(e));m(n[e]===t[e]||!i,()=>`Error creating a new Tensor. Inferred shape (${n}) does not match the provided shape (${t}). `)}}return!Or(e)&&!Array.isArray(e)&&(e=[e]),t||=n,e=r===`string`?kr(e,[],!0):Cr(e,r),H.makeTensor(e,t,r)}
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Si(e,t,n){return xi(e,t,gi(e,n),n)}var Ci=class e{static join(t){return new e(t).slice()}constructor(e){if(this.shards=[],this.previousShardIndex=0,e==null||(e instanceof Array||(e=[e]),e=e.map(e=>Or(e)?e.buffer:e),e.length===0))return;this.bufferUniformSize=e[0].byteLength;let t=0;for(let n=0;n<e.length;n++){let r=e[n];n!==e.length-1&&r.byteLength!==this.bufferUniformSize&&(this.bufferUniformSize=void 0);let i=t+r.byteLength;this.shards.push({buffer:r,start:t,end:i}),t=i}this.shards.length===0&&(this.byteLength=0),this.byteLength=this.shards[this.shards.length-1].end}slice(e=0,t=this.byteLength){if(this.shards.length===0||(e=isNaN(Number(e))?0:e,t=isNaN(Number(t))?0:t,e=Math.max(0,e),t=Math.min(this.byteLength,t),t<=e))return new ArrayBuffer(0);let n=this.findShardForByte(e);if(n===-1)throw Error(`Could not find start shard for byte ${e}`);let r=t-e,i=new ArrayBuffer(r),a=new Uint8Array(i),o=0;for(let r=n;r<this.shards.length;r++){let n=this.shards[r],i=e+o-n.start,s=o,c=Math.min(t,n.end)-n.start,l=new Uint8Array(n.buffer,i,c-i);if(a.set(l,s),o+=l.length,t<n.end)break}return i}findShardForByte(e){if(this.shards.length===0||e<0||e>=this.byteLength)return-1;if(this.bufferUniformSize!=null)return this.previousShardIndex=Math.floor(e/this.bufferUniformSize),this.previousShardIndex;function t(t){return e<t.start?-1:+(e>=t.end)}if(t(this.shards[this.previousShardIndex])===0)return this.previousShardIndex;let n=wi(this.shards,t);return n===-1?-1:(this.previousShardIndex=n,this.previousShardIndex)}};function wi(e,t){let n=0,r=e.length;for(;n<=r;){let i=Math.floor((r-n)/2)+n,a=t(e[i]);if(a===0)return i;a<0?r=i:n=i+1}return-1}
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Ti(){return H}function Ei(e,t){return H.tidy(e,t)}function Di(e){ai(e).forEach(e=>e.dispose())}function Oi(e){return H.keep(e)}function ki(e){return H.setBackend(e)}function Ai(){return H.ready()}function ji(){return H.backendName}function Mi(e,t,n=1){return H.registerBackend(e,t,n)}
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let Ni=typeof Buffer<`u`&&(typeof Blob>`u`||typeof atob>`u`||typeof btoa>`u`);function Pi(e){return Ni?Buffer.byteLength(e,`utf8`):new Blob([e]).size}function Fi(e){if(Ni)return Buffer.from(e).toString(`base64`);let t=new Uint8Array(e),n=``;for(let e=0,r=t.length;e<r;e++)n+=String.fromCharCode(t[e]);return btoa(n)}function Ii(e){if(Ni){let t=Buffer.from(e,`base64`);return t.buffer.slice(t.byteOffset,t.byteOffset+t.byteLength)}let t=atob(e),n=new Uint8Array(t.length);for(let e=0;e<t.length;++e)n.set([t.charCodeAt(e)],e);return n.buffer}function Li(e){if(e.modelTopology instanceof ArrayBuffer)throw Error(`Expected JSON model topology, received ArrayBuffer.`);return{dateSaved:new Date,modelTopologyType:`JSON`,modelTopologyBytes:e.modelTopology==null?0:Pi(JSON.stringify(e.modelTopology)),weightSpecsBytes:e.weightSpecs==null?0:Pi(JSON.stringify(e.weightSpecs)),weightDataBytes:e.weightData==null?0:new Ci(e.weightData).byteLength}}
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
var Ri=class e{constructor(){this.saveRouters=[],this.loadRouters=[]}static getInstance(){return e.instance??=new e,e.instance}static registerSaveRouter(t){e.getInstance().saveRouters.push(t)}static registerLoadRouter(t){e.getInstance().loadRouters.push(t)}static getSaveHandlers(t){return e.getHandlers(t,`save`)}static getLoadHandlers(t,n){return e.getHandlers(t,`load`,n)}static getHandlers(t,n,r){let i=[];return(n===`load`?e.getInstance().loadRouters:e.getInstance().saveRouters).forEach(e=>{let n=e(t,r);n!==null&&i.push(n)}),i}};
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let zi=`tensorflowjs`,Bi=`models_store`,Vi=`model_info_store`;function Hi(){if(!z().getBool(`IS_BROWSER`))throw Error(`Failed to obtain IndexedDB factory because the current environmentis not a web browser.`);let e=typeof window>`u`?self:window,t=e.indexedDB||e.mozIndexedDB||e.webkitIndexedDB||e.msIndexedDB||e.shimIndexedDB;if(t==null)throw Error(`The current browser does not appear to support IndexedDB.`);return t}function Ui(e){let t=e.result;t.createObjectStore(Bi,{keyPath:`modelPath`}),t.createObjectStore(Vi,{keyPath:`modelPath`})}var Wi=class{constructor(e){if(this.indexedDB=Hi(),e==null||!e)throw Error(`For IndexedDB, modelPath must not be null, undefined or empty.`);this.modelPath=e}async save(e){if(e.modelTopology instanceof ArrayBuffer)throw Error(`BrowserLocalStorage.save() does not support saving model topology in binary formats yet.`);return this.databaseAction(this.modelPath,e)}async load(){return this.databaseAction(this.modelPath)}databaseAction(e,t){return new Promise((e,n)=>{let r=this.indexedDB.open(zi,1);r.onupgradeneeded=()=>Ui(r),r.onsuccess=()=>{let i=r.result;if(t==null){let t=i.transaction(Bi,`readonly`),r=t.objectStore(Bi).get(this.modelPath);r.onsuccess=()=>{if(r.result==null)return i.close(),n(Error(`Cannot find model with path '${this.modelPath}' in IndexedDB.`));e(r.result.modelArtifacts)},r.onerror=e=>(i.close(),n(r.error)),t.oncomplete=()=>i.close()}else{t.weightData=Ci.join(t.weightData);let r=Li(t),a=i.transaction(Vi,`readwrite`),o=a.objectStore(Vi),s;try{s=o.put({modelPath:this.modelPath,modelArtifactsInfo:r})}catch(e){return n(e)}let c;s.onsuccess=()=>{c=i.transaction(Bi,`readwrite`);let s=c.objectStore(Bi),l;try{l=s.put({modelPath:this.modelPath,modelArtifacts:t,modelArtifactsInfo:r})}catch(e){return n(e)}l.onsuccess=()=>e({modelArtifactsInfo:r}),l.onerror=e=>{o=a.objectStore(Vi);let t=o.delete(this.modelPath);t.onsuccess=()=>(i.close(),n(l.error)),t.onerror=e=>(i.close(),n(l.error))}},s.onerror=e=>(i.close(),n(s.error)),a.oncomplete=()=>{c==null?i.close():c.oncomplete=()=>i.close()}}},r.onerror=e=>n(r.error)})}};Wi.URL_SCHEME=`indexeddb://`;let Gi=e=>z().getBool(`IS_BROWSER`)&&!Array.isArray(e)&&e.startsWith(Wi.URL_SCHEME)?Ki(e.slice(Wi.URL_SCHEME.length)):null;Ri.registerSaveRouter(Gi),Ri.registerLoadRouter(Gi);function Ki(e){return new Wi(e)}function qi(e){return e.startsWith(Wi.URL_SCHEME)?e.slice(Wi.URL_SCHEME.length):e}var Ji=class{constructor(){this.indexedDB=Hi()}async listModels(){return new Promise((e,t)=>{let n=this.indexedDB.open(zi,1);n.onupgradeneeded=()=>Ui(n),n.onsuccess=()=>{let r=n.result,i=r.transaction(Vi,`readonly`),a=i.objectStore(Vi).getAll();a.onsuccess=()=>{let t={};for(let e of a.result)t[e.modelPath]=e.modelArtifactsInfo;e(t)},a.onerror=e=>(r.close(),t(a.error)),i.oncomplete=()=>r.close()},n.onerror=e=>t(n.error)})}async removeModel(e){return e=qi(e),new Promise((t,n)=>{let r=this.indexedDB.open(zi,1);r.onupgradeneeded=()=>Ui(r),r.onsuccess=()=>{let i=r.result,a=i.transaction(Vi,`readwrite`),o=a.objectStore(Vi),s=o.get(e),c;s.onsuccess=()=>{if(s.result==null)return i.close(),n(Error(`Cannot find model with path '${e}' in IndexedDB.`));{let r=o.delete(e),a=()=>{c=i.transaction(Bi,`readwrite`);let r=c.objectStore(Bi).delete(e);r.onsuccess=()=>t(s.result.modelArtifactsInfo),r.onerror=e=>n(s.error)};r.onsuccess=a,r.onerror=e=>(a(),i.close(),n(s.error))}},s.onerror=e=>(i.close(),n(s.error)),a.oncomplete=()=>{c==null?i.close():c.oncomplete=()=>i.close()}},r.onerror=e=>n(r.error)})}};
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let Yi=`tensorflowjs_models`;function Xi(e){return{info:[Yi,e,`info`].join(`/`),topology:[Yi,e,`model_topology`].join(`/`),weightSpecs:[Yi,e,`weight_specs`].join(`/`),weightData:[Yi,e,`weight_data`].join(`/`),modelMetadata:[Yi,e,`model_metadata`].join(`/`)}}function Zi(e){for(let t of Object.values(e))window.localStorage.removeItem(t)}function Qi(e){let t=e.split(`/`);if(t.length<3)throw Error(`Invalid key format: ${e}`);return t.slice(1,t.length-1).join(`/`)}function $i(e){return e.startsWith(ea.URL_SCHEME)?e.slice(ea.URL_SCHEME.length):e}var ea=class{constructor(e){if(!z().getBool(`IS_BROWSER`)||typeof window>`u`||window.localStorage===void 0)throw Error(`The current environment does not support local storage.`);if(this.LS=window.localStorage,e==null||!e)throw Error(`For local storage, modelPath must not be null, undefined or empty.`);this.modelPath=e,this.keys=Xi(this.modelPath)}async save(e){if(e.modelTopology instanceof ArrayBuffer)throw Error(`BrowserLocalStorage.save() does not support saving model topology in binary formats yet.`);{let t=JSON.stringify(e.modelTopology),n=JSON.stringify(e.weightSpecs),r=Li(e),i=Ci.join(e.weightData);try{this.LS.setItem(this.keys.info,JSON.stringify(r)),this.LS.setItem(this.keys.topology,t),this.LS.setItem(this.keys.weightSpecs,n),this.LS.setItem(this.keys.weightData,Fi(i));let a={format:e.format,generatedBy:e.generatedBy,convertedBy:e.convertedBy,signature:e.signature==null?void 0:e.signature,userDefinedMetadata:e.userDefinedMetadata==null?void 0:e.userDefinedMetadata,modelInitializer:e.modelInitializer==null?void 0:e.modelInitializer,initializerSignature:e.initializerSignature==null?void 0:e.initializerSignature,trainingConfig:e.trainingConfig==null?void 0:e.trainingConfig};return this.LS.setItem(this.keys.modelMetadata,JSON.stringify(a)),{modelArtifactsInfo:r}}catch{throw Zi(this.keys),Error(`Failed to save model '${this.modelPath}' to local storage: size quota being exceeded is a possible cause of this failure: modelTopologyBytes=${r.modelTopologyBytes}, weightSpecsBytes=${r.weightSpecsBytes}, weightDataBytes=${r.weightDataBytes}.`)}}}async load(){let e=JSON.parse(this.LS.getItem(this.keys.info));if(e==null)throw Error(`In local storage, there is no model with name '${this.modelPath}'`);if(e.modelTopologyType!==`JSON`)throw Error(`BrowserLocalStorage does not support loading non-JSON model topology yet.`);let t={},n=JSON.parse(this.LS.getItem(this.keys.topology));if(n==null)throw Error(`In local storage, the topology of model '${this.modelPath}' is missing.`);t.modelTopology=n;let r=JSON.parse(this.LS.getItem(this.keys.weightSpecs));if(r==null)throw Error(`In local storage, the weight specs of model '${this.modelPath}' are missing.`);t.weightSpecs=r;let i=this.LS.getItem(this.keys.modelMetadata);if(i!=null){let e=JSON.parse(i);t.format=e.format,t.generatedBy=e.generatedBy,t.convertedBy=e.convertedBy,e.signature!=null&&(t.signature=e.signature),e.userDefinedMetadata!=null&&(t.userDefinedMetadata=e.userDefinedMetadata),e.modelInitializer!=null&&(t.modelInitializer=e.modelInitializer),e.initializerSignature!=null&&(t.initializerSignature=e.initializerSignature),e.trainingConfig!=null&&(t.trainingConfig=e.trainingConfig)}let a=this.LS.getItem(this.keys.weightData);if(a==null)throw Error(`In local storage, the binary weight values of model '${this.modelPath}' are missing.`);return t.weightData=Ii(a),t}};ea.URL_SCHEME=`localstorage://`;let ta=e=>z().getBool(`IS_BROWSER`)&&!Array.isArray(e)&&e.startsWith(ea.URL_SCHEME)?na(e.slice(ea.URL_SCHEME.length)):null;Ri.registerSaveRouter(ta),Ri.registerLoadRouter(ta);function na(e){return new ea(e)}var ra=class{constructor(){m(z().getBool(`IS_BROWSER`),()=>`Current environment is not a web browser`),m(typeof window>`u`||window.localStorage!==void 0,()=>`Current browser does not appear to support localStorage`),this.LS=window.localStorage}async listModels(){let e={};for(let t=0;t<this.LS.length;++t){let n=this.LS.key(t);if(n.startsWith(`tensorflowjs_models/`)&&n.endsWith(`/info`)){let t=Qi(n);e[t]=JSON.parse(this.LS.getItem(n))}}return e}async removeModel(e){e=$i(e);let t=Xi(e);if(this.LS.getItem(t.info)==null)throw Error(`Cannot find model at path '${e}'`);let n=JSON.parse(this.LS.getItem(t.info));return Zi(t),n}},ia=class e{constructor(){
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
this.managers={}}static getInstance(){return e.instance??=new e,e.instance}static registerManager(t,n){m(t!=null,()=>`scheme must not be undefined or null.`),t.endsWith(`://`)&&(t=t.slice(0,t.indexOf(`://`))),m(t.length>0,()=>`scheme must not be an empty string.`);let r=e.getInstance();m(r.managers[t]==null,()=>`A model store manager is already registered for scheme '${t}'.`),r.managers[t]=n}static getManager(t){let n=e.getInstance().managers[t];if(n==null)throw Error(`Cannot find model manager for scheme '${t}'`);return n}static getSchemes(){return Object.keys(e.getInstance().managers)}},aa=class{constructor(){
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
this.messageName=`setTimeoutCustom`,this.functionRefs=[],this.handledMessageCount=0,this.hasEventListener=!1}fetch(e,t){return fetch(e,t)}now(){return performance.now()}encode(e,t){if(t!==`utf-8`&&t!==`utf8`)throw Error(`Browser's encoder only supports utf-8, but got ${t}`);return this.textEncoder??=new TextEncoder,this.textEncoder.encode(e)}decode(e,t){return new TextDecoder(t).decode(e)}setTimeoutCustom(e,t){if(typeof window>`u`||!z().getBool(`USE_SETTIMEOUTCUSTOM`)){setTimeout(e,t);return}this.functionRefs.push(e),setTimeout(()=>{window.postMessage({name:this.messageName,index:this.functionRefs.length-1},`*`)},t),this.hasEventListener||(this.hasEventListener=!0,window.addEventListener(`message`,e=>{if(e.source===window&&e.data.name===this.messageName){e.stopPropagation();let t=this.functionRefs[e.data.index];t(),this.handledMessageCount++,this.handledMessageCount===this.functionRefs.length&&(this.functionRefs=[],this.handledMessageCount=0)}},!0))}isTypedArray(e){return rr(e)}};if(z().get(`IS_BROWSER`)){z().setPlatform(`browser`,new aa);try{ia.registerManager(ea.URL_SCHEME,new ra)}catch{}try{ia.registerManager(Wi.URL_SCHEME,new Ji)}catch{}}var oa=o(((e,t)=>{t.exports={}}));
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let sa={importFetch:()=>oa()},ca;var la=class{constructor(){this.util=oa(),this.textEncoder=new this.util.TextEncoder}fetch(e,t){return z().global.fetch==null?(ca??=sa.importFetch(),ca(e,t)):z().global.fetch(e,t)}now(){let e=process.hrtime();return e[0]*1e3+e[1]/1e6}encode(e,t){if(t!==`utf-8`&&t!==`utf8`)throw Error(`Node built-in encoder only supports utf-8, but got ${t}`);return this.textEncoder.encode(e)}decode(e,t){return e.length===0?``:new this.util.TextDecoder(t).decode(e)}isTypedArray(e){return this.util.types.isFloat32Array(e)||this.util.types.isInt32Array(e)||this.util.types.isUint8Array(e)||this.util.types.isUint8ClampedArray(e)}};z().get(`IS_NODE`)&&!z().get(`IS_BROWSER`)&&z().setPlatform(`node`,new la);
/**
* @license
* Copyright 2020 Google Inc. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function W(e,t=`float32`,n){return t||=`float32`,L(e),new Vr(e,t,n)}
/**
* @license
* Copyright 2020 Google Inc. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function ua(e,t){let n=U(e,`x`,`cast`);if(!E(t))throw Error(`Failed to cast to unknown dtype ${t}`);if(t===`string`&&n.dtype!==`string`||t!==`string`&&n.dtype===`string`)throw Error(`Only strings can be casted to strings`);let r={x:n},i={dtype:t};return H.runKernel(Le,r,i)}let da=bi({cast_:ua});
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function fa(e){let t={x:U(e,`x`,`clone`,`string_or_numeric`)};return H.runKernel(_t,t)}let pa=bi({clone_:fa});
/**
* @license
* Copyright 2020 Google Inc. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function ma(e,t=!1){console.log(e.toString(t))}
/**
* @license
* Copyright 2020 Google Inc. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
fi(),Gr({buffer:W,cast:da,clone:pa,print:ma});
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function ha(e,t){let n=U(e,`a`,`add`),r=U(t,`b`,`add`);[n,r]=ii(n,r);let i={a:n,b:r};return H.runKernel(`Add`,i)}let G=bi({add_:ha});
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function ga(e,t){let n=U(e,`a`,`floorDiv`),r=U(t,`b`,`floorDiv`);[n,r]=ii(n,r);let i={a:n,b:r};return H.runKernel(dt,i)}let _a=bi({floorDiv_:ga});
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function va(e,t){let n=U(e,`a`,`div`),r=U(t,`b`,`div`);if([n,r]=ii(n,r),n.dtype===`int32`&&r.dtype===`int32`)return _a(n,r);let i={a:n,b:r};return H.runKernel(rt,i,{})}let ya=bi({div_:va});
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function ba(e,t){let n=U(e,`a`,`mul`),r=U(t,`b`,`mul`);[n,r]=ii(n,r);let i={a:n,b:r};return H.runKernel(Bt,i)}let K=bi({mul_:ba});
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function xa(e){let t=U(e,`x`,`abs`);if(t.dtype===`complex64`){let e={x:t};return H.runKernel(Be,e)}{let e={x:t};return H.runKernel(`Abs`,e)}}let Sa=bi({abs_:xa});
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Ca(e,t,n,r,i=`NHWC`,a){let o=e[3];return Ea(e,[...t,o],n,a,r,null,null,Ba(i))}function wa(e,t,n,r,i,a,o=`channelsLast`){let[s,c]=ja(t),l;if(o===`channelsLast`)l=[s,c,e[3],e[3]];else if(o===`channelsFirst`)l=[s,c,e[1],e[1]];else throw Error(`Unknown dataFormat ${o}`);return Ea(e,l,n,r,i,a,!1,o)}function Ta(e,t,n,r,i,a,o=`NDHWC`){let[s,c,l]=Ma(t),u,d;if(o===`NDHWC`)d=`channelsLast`,u=[s,c,l,e[4],e[4]];else if(o===`NCDHW`)d=`channelsFirst`,u=[s,c,l,e[1],e[1]];else throw Error(`Unknown dataFormat ${o}`);return Da(e,u,n,r,i,!1,d,a)}function Ea(e,t,n,r,i,a,o=!1,s=`channelsLast`){let[c,l,u,d]=[-1,-1,-1,-1];if(s===`channelsLast`)[c,l,u,d]=e;else if(s===`channelsFirst`)[c,d,l,u]=e;else throw Error(`Unknown dataFormat ${s}`);let[f,p,,m]=t,[h,g]=ja(n),[_,v]=ja(r),y=Na(f,_),b=Na(p,v),{padInfo:x,outHeight:S,outWidth:C}=Pa(i,l,u,h,g,y,b,a,s),w=o?m*d:m,T;return s===`channelsFirst`?T=[c,w,S,C]:s===`channelsLast`&&(T=[c,S,C,w]),{batchSize:c,dataFormat:s,inHeight:l,inWidth:u,inChannels:d,outHeight:S,outWidth:C,outChannels:w,padInfo:x,strideHeight:h,strideWidth:g,filterHeight:f,filterWidth:p,effectiveFilterHeight:y,effectiveFilterWidth:b,dilationHeight:_,dilationWidth:v,inShape:e,outShape:T,filterShape:t}}function Da(e,t,n,r,i,a=!1,o=`channelsLast`,s){let[c,l,u,d,f]=[-1,-1,-1,-1,-1];if(o===`channelsLast`)[c,l,u,d,f]=e;else if(o===`channelsFirst`)[c,f,l,u,d]=e;else throw Error(`Unknown dataFormat ${o}`);let[p,m,h,,g]=t,[_,v,y]=Ma(n),[b,x,S]=Ma(r),C=Na(p,b),w=Na(m,x),T=Na(h,S),{padInfo:E,outDepth:D,outHeight:O,outWidth:k}=Fa(i,l,u,d,_,v,y,C,w,T,s),A=a?g*f:g,j;return o===`channelsFirst`?j=[c,A,D,O,k]:o===`channelsLast`&&(j=[c,D,O,k,A]),{batchSize:c,dataFormat:o,inDepth:l,inHeight:u,inWidth:d,inChannels:f,outDepth:D,outHeight:O,outWidth:k,outChannels:A,padInfo:E,strideDepth:_,strideHeight:v,strideWidth:y,filterDepth:p,filterHeight:m,filterWidth:h,effectiveFilterDepth:C,effectiveFilterHeight:w,effectiveFilterWidth:T,dilationDepth:b,dilationHeight:x,dilationWidth:S,inShape:e,outShape:j,filterShape:t}}function Oa(e,t,n,r,i){r??=Aa(e,t,n);let a=e[0],o=e[1];return[Ia((a-t+2*r)/n+1,i),Ia((o-t+2*r)/n+1,i)]}function ka(e,t,n,r,i,a){i??=Aa(e,t[0],r[0]);let o=[0,0,0,n];for(let n=0;n<3;n++)e[n]+2*i>=t[n]&&(o[n]=Ia((e[n]-t[n]+2*i)/r[n]+1,a));return o}function Aa(e,t,n,r=1){let i=Na(t,r);return Math.floor((e[0]*(n-1)-n+i)/2)}function ja(e){return typeof e==`number`?[e,e,e]:e.length===2?[e[0],e[1],1]:e}function Ma(e){return typeof e==`number`?[e,e,e]:e}function Na(e,t){return t<=1?e:e+(e-1)*(t-1)}function Pa(e,t,n,r,i,a,o,s,c){let l,u,d;if(typeof e==`number`){l={top:e,bottom:e,left:e,right:e,type:e===0?`VALID`:`NUMBER`};let i=Oa([t,n],a,r,e,s);u=i[0],d=i[1]}else if(e===`same`){u=Math.ceil(t/r),d=Math.ceil(n/i);let e=Math.max(0,(u-1)*r+a-t),s=Math.max(0,(d-1)*i+o-n),c=Math.floor(e/2),f=e-c,p=Math.floor(s/2);l={top:c,bottom:f,left:p,right:s-p,type:`SAME`}}else if(e===`valid`)l={top:0,bottom:0,left:0,right:0,type:`VALID`},u=Math.ceil((t-a+1)/r),d=Math.ceil((n-o+1)/i);else if(typeof e==`object`){let f=c===`channelsLast`?e[1][0]:e[2][0],p=c===`channelsLast`?e[1][1]:e[2][1],m=c===`channelsLast`?e[2][0]:e[3][0],h=c===`channelsLast`?e[2][1]:e[3][1];l={top:f,bottom:p,left:m,right:h,type:f===0&&p===0&&m===0&&h===0?`VALID`:`EXPLICIT`},u=Ia((t-a+f+p)/r+1,s),d=Ia((n-o+m+h)/i+1,s)}else throw Error(`Unknown padding parameter: ${e}`);return{padInfo:l,outHeight:u,outWidth:d}}function Fa(e,t,n,r,i,a,o,s,c,l,u){let d,f,p,m;if(e===`valid`&&(e=0),typeof e==`number`){d={top:e,bottom:e,left:e,right:e,front:e,back:e,type:e===0?`VALID`:`NUMBER`};let h=ka([t,n,r,1],[s,c,l],1,[i,a,o],e,u);f=h[0],p=h[1],m=h[2]}else if(e===`same`){f=Math.ceil(t/i),p=Math.ceil(n/a),m=Math.ceil(r/o);let e=(f-1)*i+s-t,u=(p-1)*a+c-n,h=(m-1)*o+l-r,g=Math.floor(e/2),_=e-g,v=Math.floor(u/2),y=u-v,b=Math.floor(h/2);d={top:v,bottom:y,left:b,right:h-b,front:g,back:_,type:`SAME`}}else throw Error(`Unknown padding parameter: ${e}`);return{padInfo:d,outDepth:f,outHeight:p,outWidth:m}}function Ia(e,t){if(!t)return Math.trunc(e);switch(t){case`round`:return Math.round(e);case`ceil`:return Math.ceil(e);case`floor`:return Math.floor(e);default:throw Error(`Unknown roundingMode ${t}`)}}function La(e){let[t,n,r]=ja(e);return t===1&&n===1&&r===1}function Ra(e,t){return La(e)||La(t)}function za(e){return ja(e).every(e=>e>0)}function Ba(e){if(e===`NHWC`)return`channelsLast`;if(e===`NCHW`)return`channelsFirst`;throw Error(`Unknown dataFormat ${e}`)}function Va(e,t,n){if(n!=null){if(typeof t==`string`)throw Error(`Error in ${e}: pad must be an integer when using dimRoundingMode ${n} but got pad ${t}.`);if(typeof t==`number`)m(y(t),()=>`Error in ${e}: pad must be an integer when using dimRoundingMode ${n} but got pad ${t}.`);else if(typeof t==`object`)t.forEach(t=>{t.forEach(t=>{m(y(t),()=>`Error in ${e}: pad must be an integer when using dimRoundingMode ${n} but got pad ${t}.`)})});else throw Error(`Error in ${e}: Unknown padding parameter: ${t}`)}}
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Ha(e,t){let n={x:U(e,`x`,`reshape`,`string_or_numeric`)},r={shape:t};return H.runKernel(en,n,r)}let Ua=bi({reshape_:Ha});
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Wa(e,t=0){m(e.length>=1,()=>`Pass at least one tensor to concat`);let n=yi(e,`tensors`,`concat`,`string_or_numeric`);if(n[0].dtype===`complex64`&&n.forEach(e=>{if(e.dtype!==`complex64`)throw Error(`Cannot concatenate complex64 tensors with a tensor
          with dtype ${e.dtype}. `)}),n.length===1)return pa(n[0]);let r=n,i={axis:t};return H.runKernel(Ve,r,i)}let Ga=bi({concat_:Wa});
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Ka(e,t,n=!1,r=!1){let i=U(e,`a`,`matMul`),a=U(t,`b`,`matMul`);[i,a]=ii(i,a);let o={a:i,b:a},s={transposeA:n,transposeB:r};return H.runKernel(Me,o,s)}let qa=bi({matMul_:Ka});
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Ja(e){let t={x:U(e,`x`,`tanh`,`float32`)};return H.runKernel(In,t)}let Ya=bi({tanh_:Ja});
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Xa(e,t){let n=U(e,`broadcastTo`,`x`),r=n.shape;if(L(t),t.length<n.rank)throw Error(`broadcastTo(): shape.length=${t.length} < input.rank=${n.rank}.`);if(t.length>n.rank){let e=n.shape.slice();for(;e.length<t.length;)e.unshift(1);n=Ua(n,e)}let i=n.shape,a=Array.from(t);for(let e=t.length-1;e>=0;e--)if(i[e]===t[e])a[e]=1;else if(n.shape[e]!==1)throw Error(`broadcastTo(): [${r}] cannot be broadcast to [${t}].`);if(a.map((e,t)=>e>1?t:-1).filter(e=>e>=0).length===0)return pa(n);let o={x:n},s={reps:a};return H.runKernel(Ln,o,s)}let Za=bi({broadcastTo_:Xa});
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Qa(e,t,n){L(e),n||=N(t);let r={shape:e,value:t,dtype:n};return H.runKernel(ct,{},r)}
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function $a(e,t,n,r,i=`NHWC`,a=[1,1],o){let s=U(e,`x`,`conv2d`,`float32`),c=U(t,`filter`,`conv2d`,`float32`),l=s,u=!1;s.rank===3&&(u=!0,l=Ua(s,[1,s.shape[0],s.shape[1],s.shape[2]])),m(l.rank===4,()=>`Error in conv2d: input must be rank 4, but got rank ${l.rank}.`),m(c.rank===4,()=>`Error in conv2d: filter must be rank 4, but got rank ${c.rank}.`),Va(`conv2d`,r,o);let d=i===`NHWC`?l.shape[3]:l.shape[1];m(d===c.shape[2],()=>`Error in conv2d: depth of input (${d}) must match input depth for filter ${c.shape[2]}.`),m(Ra(n,a),()=>`Error in conv2D: Either strides or dilations must be 1. Got strides ${n} and dilations '${a}'`),m(za(a),()=>`Error in conv2D: Dilated rates should be larger than 0.`),m(za(n),()=>`Error in conv2D: Strides should be larger than 0.`);let f={x:l,filter:c},p={strides:n,pad:r,dataFormat:i,dilations:a,dimRoundingMode:o},h=H.runKernel(He,f,p);return u?Ua(h,[h.shape[1],h.shape[2],h.shape[3]]):h}let eo=bi({conv2d_:$a});
/**
* @license
* Copyright 2017 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function to(e,t){let n=e.length,r=[];for(let i=0;i<n;i++){let a=n-1-i,o=e[a]||1;(t[t.length-1-i]||1)>1&&o===1&&r.unshift(a)}return r}function no(e,t){let n=Math.max(e.length,t.length),r=Array(n);for(let i=0;i<n;i++){let a=e[e.length-i-1];a??=1;let o=t[t.length-i-1];if(o??=1,a===1)r[n-i-1]=o;else if(o===1)r[n-i-1]=a;else if(a!==o){let n=`Operands could not be broadcast together with shapes ${e} and ${t}.`;throw Error(n)}else r[n-i-1]=a}return r}
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function ro(e){let t={x:U(e,`x`,`zerosLike`)};return H.runKernel(Un,t)}let io=bi({zerosLike_:ro});
/**
* @license
* Copyright 2017 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function ao(e,t){for(let n=0;n<e.length;++n)if(e[e.length-n-1]!==t-1-n)return!1;return!0}function oo(e,t,n){let r=e.length+t.length,i=[],a=0,o=0;for(let s=0;s<r;s++)n.indexOf(s)===-1?i.push(e[a++]):i.push(t[o++]);return i}function so(e,t){let n=[],r=e.length;for(let i=0;i<r;i++)t.indexOf(i)===-1&&n.push(e[i]);return[n,t.map(t=>e[t])]}function co(e,t){return oo(e,t.map(e=>1),t)}function lo(e,t,n){m(ao(t,n),()=>`${e} supports only inner-most axes for now. Got axes ${t} and rank-${n} input.`)}function uo(e,t){if(ao(e,t))return null;let n=[];for(let r=0;r<t;++r)e.indexOf(r)===-1&&n.push(r);return e.forEach(e=>n.push(e)),n}function fo(e){return e.map((e,t)=>[t,e]).sort((e,t)=>e[1]-t[1]).map(e=>e[0])}function po(e,t){let n=[];for(let r=t-e;r<t;++r)n.push(r);return n}
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function mo(e,t=null,n=!1){let r={x:U(e,`x`,`max`)},i={reductionIndices:t,keepDims:n};return H.runKernel(`Max`,r,i)}let ho=bi({max_:mo});
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function go(e,t){let n=U(e,`base`,`pow`),r=U(t,`exp`,`pow`);[n,r]=ii(n,r);let i={a:n,b:r};return H.runKernel(`Pow`,i)}let _o=bi({pow_:go});
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function vo(e,t){if((Or(e)&&t!==`string`||Array.isArray(e))&&t!==`complex64`)throw Error(`Error creating a new Scalar: value must be a primitive (number|boolean|string)`);if(t===`string`&&Or(e)&&!(e instanceof Uint8Array))throw Error("When making a scalar from encoded string, the value must be `Uint8Array`.");return xi(e,[],[],t)}
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function yo(e){let t={x:U(e,`x`,`sqrt`,`float32`)};return H.runKernel(bn,t)}let bo=bi({sqrt_:yo});
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function xo(e){let t=U(e,`x`,`square`);return H.runKernel(`Square`,{x:t},{})}let So=bi({square_:xo});
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Co(e,t){m(ee(e),()=>`The f passed in variableGrads(f) must be a function`),m(t==null||Array.isArray(t)&&t.every(e=>e instanceof Jr),()=>`The varList passed in variableGrads(f, varList) must be an array of variables`);let n=t!=null;if(!n){t=[];for(let e in H.registeredVariables)t.push(H.registeredVariables[e])}let r=n?t.filter(e=>!e.trainable):null,i=t.length;t=t.filter(e=>e.trainable),m(t.length>0,()=>`variableGrads() expects at least one of the input variables to be trainable, but none of the ${i} variables is trainable.`);let{value:a,grads:o}=H.gradients(e,t,null,!0);m(o.some(e=>e!=null),()=>`Cannot find a connection between any variable and the result of the loss function y=f(x). Please make sure the operations that use variables are inside the function f passed to minimize().`),m(a.rank===0,()=>`The f passed in variableGrads(f) must return a scalar, but it returned a rank-${a.rank} tensor`);let s={};return t.forEach((e,t)=>{o[t]!=null&&(s[e.name]=o[t])}),r?.forEach(e=>s[e.name]=null),{value:a,grads:s}}
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function wo(e){let t={x:U(e,`x`,`softplus`)};return H.runKernel(yn,t)}let To=bi({softplus_:wo});
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Eo(e,t){let n=U(e,`a`,`sub`),r=U(t,`b`,`sub`);[n,r]=ii(n,r);let i={a:n,b:r};return H.runKernel(`Sub`,i)}let Do=bi({sub_:Eo});
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Oo(e,t){let n=U(e,`a`,`maximum`),r=U(t,`b`,`maximum`);[n,r]=ii(n,r),n.dtype===`bool`&&(n=da(n,`int32`),r=da(r,`int32`)),no(n.shape,r.shape);let i={a:n,b:r};return H.runKernel(At,i)}let ko=bi({maximum_:Oo});
/**
* @license
* Copyright 2020 Google Inc. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Ao(e,t=null,n=!1){let r={x:U(e,`x`,`mean`)},i={axis:t,keepDims:n};return H.runKernel(It,r,i)}let jo=bi({mean_:Ao});var Mo=o(((e,t)=>{(function(e,t,n){function r(e){var t=this,n=o();t.next=function(){var e=2091639*t.s0+t.c*23283064365386963e-26;return t.s0=t.s1,t.s1=t.s2,t.s2=e-(t.c=e|0)},t.c=1,t.s0=n(` `),t.s1=n(` `),t.s2=n(` `),t.s0-=n(e),t.s0<0&&(t.s0+=1),t.s1-=n(e),t.s1<0&&(t.s1+=1),t.s2-=n(e),t.s2<0&&(t.s2+=1),n=null}function i(e,t){return t.c=e.c,t.s0=e.s0,t.s1=e.s1,t.s2=e.s2,t}function a(e,t){var n=new r(e),a=t&&t.state,o=n.next;return o.int32=function(){return n.next()*4294967296|0},o.double=function(){return o()+(o()*2097152|0)*11102230246251565e-32},o.quick=o,a&&(typeof a==`object`&&i(a,n),o.state=function(){return i(n,{})}),o}function o(){var e=4022871197;return function(t){t=String(t);for(var n=0;n<t.length;n++){e+=t.charCodeAt(n);var r=.02519603282416938*e;e=r>>>0,r-=e,r*=e,e=r>>>0,r-=e,e+=r*4294967296}return(e>>>0)*23283064365386963e-26}}t&&t.exports?t.exports=a:n&&n.amd?n(function(){return a}):this.alea=a})(e,typeof t==`object`&&t,typeof define==`function`&&define)})),No=o(((e,t)=>{(function(e,t,n){function r(e){var t=this,n=``;t.x=0,t.y=0,t.z=0,t.w=0,t.next=function(){var e=t.x^t.x<<11;return t.x=t.y,t.y=t.z,t.z=t.w,t.w^=t.w>>>19^e^e>>>8},e===(e|0)?t.x=e:n+=e;for(var r=0;r<n.length+64;r++)t.x^=n.charCodeAt(r)|0,t.next()}function i(e,t){return t.x=e.x,t.y=e.y,t.z=e.z,t.w=e.w,t}function a(e,t){var n=new r(e),a=t&&t.state,o=function(){return(n.next()>>>0)/4294967296};return o.double=function(){do var e=((n.next()>>>11)+(n.next()>>>0)/4294967296)/(1<<21);while(e===0);return e},o.int32=n.next,o.quick=o,a&&(typeof a==`object`&&i(a,n),o.state=function(){return i(n,{})}),o}t&&t.exports?t.exports=a:n&&n.amd?n(function(){return a}):this.xor128=a})(e,typeof t==`object`&&t,typeof define==`function`&&define)})),Po=o(((e,t)=>{(function(e,t,n){function r(e){var t=this,n=``;t.next=function(){var e=t.x^t.x>>>2;return t.x=t.y,t.y=t.z,t.z=t.w,t.w=t.v,(t.d=t.d+362437|0)+(t.v=t.v^t.v<<4^(e^e<<1))|0},t.x=0,t.y=0,t.z=0,t.w=0,t.v=0,e===(e|0)?t.x=e:n+=e;for(var r=0;r<n.length+64;r++)t.x^=n.charCodeAt(r)|0,r==n.length&&(t.d=t.x<<10^t.x>>>4),t.next()}function i(e,t){return t.x=e.x,t.y=e.y,t.z=e.z,t.w=e.w,t.v=e.v,t.d=e.d,t}function a(e,t){var n=new r(e),a=t&&t.state,o=function(){return(n.next()>>>0)/4294967296};return o.double=function(){do var e=((n.next()>>>11)+(n.next()>>>0)/4294967296)/(1<<21);while(e===0);return e},o.int32=n.next,o.quick=o,a&&(typeof a==`object`&&i(a,n),o.state=function(){return i(n,{})}),o}t&&t.exports?t.exports=a:n&&n.amd?n(function(){return a}):this.xorwow=a})(e,typeof t==`object`&&t,typeof define==`function`&&define)})),Fo=o(((e,t)=>{(function(e,t,n){function r(e){var t=this;t.next=function(){var e=t.x,n=t.i,r=e[n],i;return r^=r>>>7,i=r^r<<24,r=e[n+1&7],i^=r^r>>>10,r=e[n+3&7],i^=r^r>>>3,r=e[n+4&7],i^=r^r<<7,r=e[n+7&7],r^=r<<13,i^=r^r<<9,e[n]=i,t.i=n+1&7,i};function n(e,t){var n,r=[];if(t===(t|0))r[0]=t;else for(t=``+t,n=0;n<t.length;++n)r[n&7]=r[n&7]<<15^t.charCodeAt(n)+r[n+1&7]<<13;for(;r.length<8;)r.push(0);for(n=0;n<8&&r[n]===0;++n);for(n==8?r[7]=-1:r[n],e.x=r,e.i=0,n=256;n>0;--n)e.next()}n(t,e)}function i(e,t){return t.x=e.x.slice(),t.i=e.i,t}function a(e,t){e??=+new Date;var n=new r(e),a=t&&t.state,o=function(){return(n.next()>>>0)/4294967296};return o.double=function(){do var e=((n.next()>>>11)+(n.next()>>>0)/4294967296)/(1<<21);while(e===0);return e},o.int32=n.next,o.quick=o,a&&(a.x&&i(a,n),o.state=function(){return i(n,{})}),o}t&&t.exports?t.exports=a:n&&n.amd?n(function(){return a}):this.xorshift7=a})(e,typeof t==`object`&&t,typeof define==`function`&&define)})),Io=o(((e,t)=>{(function(e,t,n){function r(e){var t=this;t.next=function(){var e=t.w,n=t.X,r=t.i,i,a;return t.w=e=e+1640531527|0,a=n[r+34&127],i=n[r=r+1&127],a^=a<<13,i^=i<<17,a^=a>>>15,i^=i>>>12,a=n[r]=a^i,t.i=r,a+(e^e>>>16)|0};function n(e,t){var n,r,i,a,o,s=[],c=128;for(t===(t|0)?(r=t,t=null):(t+=`\0`,r=0,c=Math.max(c,t.length)),i=0,a=-32;a<c;++a)t&&(r^=t.charCodeAt((a+32)%t.length)),a===0&&(o=r),r^=r<<10,r^=r>>>15,r^=r<<4,r^=r>>>13,a>=0&&(o=o+1640531527|0,n=s[a&127]^=r+o,i=n==0?i+1:0);for(i>=128&&(s[(t&&t.length||0)&127]=-1),i=127,a=512;a>0;--a)r=s[i+34&127],n=s[i=i+1&127],r^=r<<13,n^=n<<17,r^=r>>>15,n^=n>>>12,s[i]=r^n;e.w=o,e.X=s,e.i=i}n(t,e)}function i(e,t){return t.i=e.i,t.w=e.w,t.X=e.X.slice(),t}function a(e,t){e??=+new Date;var n=new r(e),a=t&&t.state,o=function(){return(n.next()>>>0)/4294967296};return o.double=function(){do var e=((n.next()>>>11)+(n.next()>>>0)/4294967296)/(1<<21);while(e===0);return e},o.int32=n.next,o.quick=o,a&&(a.X&&i(a,n),o.state=function(){return i(n,{})}),o}t&&t.exports?t.exports=a:n&&n.amd?n(function(){return a}):this.xor4096=a})(e,typeof t==`object`&&t,typeof define==`function`&&define)})),Lo=o(((e,t)=>{(function(e,t,n){function r(e){var t=this,n=``;t.next=function(){var e=t.b,n=t.c,r=t.d,i=t.a;return e=e<<25^e>>>7^n,n=n-r|0,r=r<<24^r>>>8^i,i=i-e|0,t.b=e=e<<20^e>>>12^n,t.c=n=n-r|0,t.d=r<<16^n>>>16^i,t.a=i-e|0},t.a=0,t.b=0,t.c=-1640531527,t.d=1367130551,e===Math.floor(e)?(t.a=e/4294967296|0,t.b=e|0):n+=e;for(var r=0;r<n.length+20;r++)t.b^=n.charCodeAt(r)|0,t.next()}function i(e,t){return t.a=e.a,t.b=e.b,t.c=e.c,t.d=e.d,t}function a(e,t){var n=new r(e),a=t&&t.state,o=function(){return(n.next()>>>0)/4294967296};return o.double=function(){do var e=((n.next()>>>11)+(n.next()>>>0)/4294967296)/(1<<21);while(e===0);return e},o.int32=n.next,o.quick=o,a&&(typeof a==`object`&&i(a,n),o.state=function(){return i(n,{})}),o}t&&t.exports?t.exports=a:n&&n.amd?n(function(){return a}):this.tychei=a})(e,typeof t==`object`&&t,typeof define==`function`&&define)})),Ro=o(((e,t)=>{(function(e,n,r){var i=256,a=6,o=52,s=`random`,c=r.pow(i,a),l=r.pow(2,o),u=l*2,d=i-1,f;function p(e,t,o){var d=[];t=t==1?{entropy:!0}:t||{};var f=_(g(t.entropy?[e,y(n)]:e??v(),3),d),p=new m(d),b=function(){for(var e=p.g(a),t=c,n=0;e<l;)e=(e+n)*i,t*=i,n=p.g(1);for(;e>=u;)e/=2,t/=2,n>>>=1;return(e+n)/t};return b.int32=function(){return p.g(4)|0},b.quick=function(){return p.g(4)/4294967296},b.double=b,_(y(p.S),n),(t.pass||o||function(e,t,n,i){return i&&(i.S&&h(i,p),e.state=function(){return h(p,{})}),n?(r[s]=e,t):e})(b,f,`global`in t?t.global:this==r,t.state)}function m(e){var t,n=e.length,r=this,a=0,o=r.i=r.j=0,s=r.S=[];for(n||(e=[n++]);a<i;)s[a]=a++;for(a=0;a<i;a++)s[a]=s[o=d&o+e[a%n]+(t=s[a])],s[o]=t;(r.g=function(e){for(var t,n=0,a=r.i,o=r.j,s=r.S;e--;)t=s[a=d&a+1],n=n*i+s[d&(s[a]=s[o=d&o+t])+(s[o]=t)];return r.i=a,r.j=o,n})(i)}function h(e,t){return t.i=e.i,t.j=e.j,t.S=e.S.slice(),t}function g(e,t){var n=[],r=typeof e,i;if(t&&r==`object`)for(i in e)try{n.push(g(e[i],t-1))}catch{}return n.length?n:r==`string`?e:e+`\0`}function _(e,t){for(var n=e+``,r,i=0;i<n.length;)t[d&i]=d&(r^=t[d&i]*19)+n.charCodeAt(i++);return y(t)}function v(){try{var t;return f&&(t=f.randomBytes)?t=t(i):(t=new Uint8Array(i),(e.crypto||e.msCrypto).getRandomValues(t)),y(t)}catch{var r=e.navigator,a=r&&r.plugins;return[+new Date,e,a,e.screen,y(n)]}}function y(e){return String.fromCharCode.apply(0,e)}if(_(r.random(),n),typeof t==`object`&&t.exports){t.exports=p;try{f=oa()}catch{}}else typeof define==`function`&&define.amd?define(function(){return p}):r[`seed`+s]=p})(typeof self<`u`?self:e,[],Math)})),zo=o(((e,t)=>{var n=Mo(),r=No(),i=Po(),a=Fo(),o=Io(),s=Lo(),c=Ro();c.alea=n,c.xor128=r,c.xorwow=i,c.xorshift7=a,c.xor4096=o,c.tychei=s,t.exports=c}));
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Bo(e){let t={x:U(e,`x`,`relu`)};return H.runKernel($t,t)}let Vo=bi({relu_:Bo});
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Ho(e,t,n){if(g(e),t!=null&&t.length!==2)throw Error(`tensor2d() requires shape to have two numbers`);let r=gi(e,n);if(r.length!==2&&r.length!==1)throw Error(`tensor2d() requires values to be number[][] or flat/TypedArray`);if(r.length===1&&t==null)throw Error("tensor2d() requires shape to be provided when `values` are a flat/TypedArray");return xi(e,t,r,n)}
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Uo(e,t,n){if(g(e),t!=null&&t.length!==4)throw Error(`tensor4d() requires shape to have four numbers`);let r=gi(e,n);if(r.length!==4&&r.length!==1)throw Error(`tensor4d() requires values to be number[][][][] or flat/TypedArray`);if(r.length===1&&t==null)throw Error("tensor4d() requires shape to be provided when `values` are a flat array");return xi(e,t,r,n)}function Wo(e,t,n){let r=t.shape.length,i=r>1?t.shape[r-1]:1,a=n.length,o=1;for(let e=i;e<a;++e)o*=n[e];let s=i<1?1:i,c=_(t.shape)/s,l=[...P(n.slice(0,i)),1],u=_(n);return{sliceRank:i,numUpdates:c,sliceSize:o,strides:l,outputSize:u}}
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Go(e,t){let n=[];for(let e=0;e<t.length;e++)t[e]&&n.push(e);let r=W(e,`int32`),i=W([n.length,e.length],`int32`);for(let t=0;t<n.length;t++){let a=r.indexToLoc(n[t]),o=t*e.length;i.values.set(a,o)}return i.toTensor()}
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Ko(e,t,n){let r=qo(e,t,n),i=r<0?-(r+1):r;e.splice(i,0,t)}function qo(e,t,n){return Yo(e,t,n||Jo)}function Jo(e,t){return e>t?1:e<t?-1:0}function Yo(e,t,n){let r=0,i=e.length,a=0,o=!1;for(;r<i;){a=r+(i-r>>>1);let s=n(t,e[a]);s>0?r=a+1:(i=a,o=!s)}return o?r:-r-1}
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Xo(e,t,n,r,i){return $o(e,t,n,r,i,0)}function Zo(e,t,n,r,i,a){return $o(e,t,n,r,i,0,!1,a,!0)}function Qo(e,t,n,r,i,a){return $o(e,t,n,r,i,a,!0)}function $o(e,t,n,r,i,a,o=!1,s=!1,c=!1){let l=[];for(let e=0;e<t.length;e++)t[e]>i&&l.push({score:t[e],boxIndex:e,suppressBeginIndex:0});l.sort(ns);let u=a>0?-.5/a:0,d=[],f=[];for(;d.length<n&&l.length>0;){let t=l.pop(),{score:n,boxIndex:a,suppressBeginIndex:o}=t;if(n<i)break;let s=!1;for(let n=d.length-1;n>=o;--n){let o=es(e,a,d[n]);if(o>=r){s=!0;break}if(t.score*=ts(r,u,o),t.score<=i)break}t.suppressBeginIndex=d.length,s||(t.score===n?(d.push(a),f.push(t.score)):t.score>i&&Ko(l,t,ns))}let p=d.length,m=n-p;s&&m>0&&(d.push(...Array(m).fill(0)),f.push(...Array(m).fill(0)));let h={selectedIndices:d};return o&&(h.selectedScores=f),c&&(h.validOutputs=p),h}function es(e,t,n){let r=e.subarray(t*4,t*4+4),i=e.subarray(n*4,n*4+4),a=Math.min(r[0],r[2]),o=Math.min(r[1],r[3]),s=Math.max(r[0],r[2]),c=Math.max(r[1],r[3]),l=Math.min(i[0],i[2]),u=Math.min(i[1],i[3]),d=Math.max(i[0],i[2]),f=Math.max(i[1],i[3]),p=(s-a)*(c-o),m=(d-l)*(f-u);if(p<=0||m<=0)return 0;let h=Math.max(a,l),g=Math.max(o,u),_=Math.min(s,d),v=Math.min(c,f),y=Math.max(_-h,0)*Math.max(v-g,0);return y/(p+m-y)}function ts(e,t,n){let r=Math.exp(t*n*n);return n<=e?r:0}function ns(e,t){return e.score-t.score||e.score===t.score&&t.boxIndex-e.boxIndex}
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let rs=new Map,is=new Map;var as=class{getClassName(){return this.constructor.className}static fromConfig(e,t){return new e(t)}},os=class e{constructor(){this.classNameMap={}}static getMap(){return e.instance??=new e,e.instance}static register(t){e.getMap().classNameMap[t.className]=[t,t.fromConfig]}};function ss(e,t,n){m(e.className!=null,()=>`Class being registered does not have the static className property defined.`),m(typeof e.className==`string`,()=>`className is required to be a string, but got type `+typeof e.className),m(e.className.length>0,()=>`Class being registered has an empty-string as its className, which is disallowed.`),t===void 0&&(t=`Custom`),n===void 0&&(n=e.className);let r=n,i=t+`>`+r;return os.register(e),rs.set(i,e),is.set(e,i),e}
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
var cs=class extends as{minimize(e,t=!1,n){let{value:r,grads:i}=this.computeGradients(e,n);if(n!=null){let e=n.map(e=>({name:e.name,tensor:i[e.name]}));this.applyGradients(e)}else this.applyGradients(i);return Di(i),t?r:(r.dispose(),null)}get iterations(){return this.iterations_??=0,this.iterations_}incrementIterations(){this.iterations_=this.iterations+1}computeGradients(e,t){return Co(e,t)}dispose(){this.iterations_!=null&&Di(this.iterations_)}async saveIterations(){return this.iterations_??=0,{name:`iter`,tensor:vo(this.iterations_,`int32`)}}async getWeights(){throw Error(`getWeights() is not implemented for this optimizer yet.`)}async setWeights(e){throw Error(`setWeights() is not implemented for this optimizer class ${this.getClassName()}`)}async extractIterations(e){return this.iterations_=(await e[0].tensor.data())[0],e.slice(1)}};Object.defineProperty(cs,Symbol.hasInstance,{value:e=>e.minimize!=null&&e.computeGradients!=null&&e.applyGradients!=null});
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
var ls=class extends cs{static get className(){return`Adadelta`}constructor(e,t,n=null){super(),this.learningRate=e,this.rho=t,this.epsilon=n,this.accumulatedGrads=[],this.accumulatedUpdates=[],n??(this.epsilon=H.backend.epsilon())}applyGradients(e){(Array.isArray(e)?e.map(e=>e.name):Object.keys(e)).forEach((t,n)=>{let r=H.registeredVariables[t];this.accumulatedGrads[n]??(this.accumulatedGrads[n]={originalName:`${t}/accum_grad`,variable:Ei(()=>io(r).variable(!1))}),this.accumulatedUpdates[n]??(this.accumulatedUpdates[n]={originalName:`${t}/accum_var`,variable:Ei(()=>io(r).variable(!1))});let i=Array.isArray(e)?e[n].tensor:e[t];if(i==null)return;let a=this.accumulatedGrads[n].variable,o=this.accumulatedUpdates[n].variable;Ei(()=>{let e=G(K(a,this.rho),K(So(i),1-this.rho)),t=K(ya(bo(G(o,this.epsilon)),bo(G(a,this.epsilon))),i),n=G(K(o,this.rho),K(So(t),1-this.rho));a.assign(e),o.assign(n);let s=G(K(t,-this.learningRate),r);r.assign(s)})}),this.incrementIterations()}dispose(){this.accumulatedUpdates!=null&&(Di(this.accumulatedGrads.map(e=>e.variable)),Di(this.accumulatedUpdates.map(e=>e.variable)))}async getWeights(){let e=[...this.accumulatedGrads,...this.accumulatedUpdates];return[await this.saveIterations()].concat(e.map(e=>({name:e.originalName,tensor:e.variable})))}async setWeights(e){e=await this.extractIterations(e);let t=e.length/2;this.accumulatedGrads=e.slice(0,t).map(e=>({originalName:e.name,variable:e.tensor.variable(!1)})),this.accumulatedUpdates=e.slice(t,t*2).map(e=>({originalName:e.name,variable:e.tensor.variable(!1)}))}getConfig(){return{learningRate:this.learningRate,rho:this.rho,epsilon:this.epsilon}}static fromConfig(e,t){return new e(t.learningRate,t.rho,t.epsilon)}},us=class extends cs{static get className(){
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
return`Adagrad`}constructor(e,t=.1){super(),this.learningRate=e,this.initialAccumulatorValue=t,this.accumulatedGrads=[]}applyGradients(e){(Array.isArray(e)?e.map(e=>e.name):Object.keys(e)).forEach((t,n)=>{let r=H.registeredVariables[t];this.accumulatedGrads[n]??(this.accumulatedGrads[n]={originalName:`${t}/accumulator`,variable:Ei(()=>Qa(r.shape,this.initialAccumulatorValue).variable(!1))});let i=Array.isArray(e)?e[n].tensor:e[t];if(i==null)return;let a=this.accumulatedGrads[n].variable;Ei(()=>{let e=G(a,So(i));a.assign(e);let t=G(K(ya(i,bo(G(e,H.backend.epsilon()))),-this.learningRate),r);r.assign(t)})}),this.incrementIterations()}dispose(){this.accumulatedGrads!=null&&Di(this.accumulatedGrads.map(e=>e.variable))}async getWeights(){return[await this.saveIterations()].concat(this.accumulatedGrads.map(e=>({name:e.originalName,tensor:e.variable})))}async setWeights(e){e=await this.extractIterations(e),this.accumulatedGrads=e.map(e=>({originalName:e.name,variable:e.tensor.variable(!1)}))}getConfig(){return{learningRate:this.learningRate,initialAccumulatorValue:this.initialAccumulatorValue}}static fromConfig(e,t){return new e(t.learningRate,t.initialAccumulatorValue)}},ds=class extends cs{static get className(){
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
return`Adam`}constructor(e,t,n,r=null){super(),this.learningRate=e,this.beta1=t,this.beta2=n,this.epsilon=r,this.accumulatedFirstMoment=[],this.accumulatedSecondMoment=[],Ei(()=>{this.accBeta1=vo(t).variable(),this.accBeta2=vo(n).variable()}),r??(this.epsilon=H.backend.epsilon())}applyGradients(e){let t=Array.isArray(e)?e.map(e=>e.name):Object.keys(e);Ei(()=>{let n=Do(1,this.accBeta1),r=Do(1,this.accBeta2);t.forEach((t,i)=>{let a=H.registeredVariables[t];this.accumulatedFirstMoment[i]??(this.accumulatedFirstMoment[i]={originalName:`${t}/m`,variable:Ei(()=>io(a).variable(!1))}),this.accumulatedSecondMoment[i]??(this.accumulatedSecondMoment[i]={originalName:`${t}/v`,variable:Ei(()=>io(a).variable(!1))});let o=Array.isArray(e)?e[i].tensor:e[t];if(o==null)return;let s=this.accumulatedFirstMoment[i].variable,c=this.accumulatedSecondMoment[i].variable,l=G(K(s,this.beta1),K(o,1-this.beta1)),u=G(K(c,this.beta2),K(So(o),1-this.beta2)),d=ya(l,n),f=ya(u,r);s.assign(l),c.assign(u);let p=G(K(ya(d,G(bo(f),this.epsilon)),-this.learningRate),a);a.assign(p)}),this.accBeta1.assign(K(this.accBeta1,this.beta1)),this.accBeta2.assign(K(this.accBeta2,this.beta2))}),this.incrementIterations()}dispose(){this.accBeta1.dispose(),this.accBeta2.dispose(),this.accumulatedFirstMoment!=null&&Di(this.accumulatedFirstMoment.map(e=>e.variable)),this.accumulatedSecondMoment!=null&&Di(this.accumulatedSecondMoment.map(e=>e.variable))}async getWeights(){let e=[...this.accumulatedFirstMoment,...this.accumulatedSecondMoment];return[await this.saveIterations()].concat(e.map(e=>({name:e.originalName,tensor:e.variable})))}async setWeights(e){e=await this.extractIterations(e),Ei(()=>{this.accBeta1.assign(_o(this.beta1,this.iterations_+1)),this.accBeta2.assign(_o(this.beta2,this.iterations_+1))});let t=e.length/2;this.accumulatedFirstMoment=e.slice(0,t).map(e=>({originalName:e.name,variable:e.tensor.variable(!1)})),this.accumulatedSecondMoment=e.slice(t,t*2).map(e=>({originalName:e.name,variable:e.tensor.variable(!1)}))}getConfig(){return{learningRate:this.learningRate,beta1:this.beta1,beta2:this.beta2,epsilon:this.epsilon}}static fromConfig(e,t){return new e(t.learningRate,t.beta1,t.beta2,t.epsilon)}},fs=class extends cs{static get className(){
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
return`Adamax`}constructor(e,t,n,r=null,i=0){super(),this.learningRate=e,this.beta1=t,this.beta2=n,this.epsilon=r,this.decay=i,this.accumulatedFirstMoment=[],this.accumulatedWeightedInfNorm=[],Ei(()=>{this.iteration=vo(0).variable(),this.accBeta1=vo(t).variable()}),r??(this.epsilon=H.backend.epsilon())}applyGradients(e){let t=Array.isArray(e)?e.map(e=>e.name):Object.keys(e);Ei(()=>{let n=Do(1,this.accBeta1),r=ya(-this.learningRate,G(K(this.iteration,this.decay),1));t.forEach((t,i)=>{let a=H.registeredVariables[t];this.accumulatedFirstMoment[i]??(this.accumulatedFirstMoment[i]={originalName:`${t}/m`,variable:io(a).variable(!1)}),this.accumulatedWeightedInfNorm[i]??(this.accumulatedWeightedInfNorm[i]={originalName:`${t}/v`,variable:io(a).variable(!1)});let o=Array.isArray(e)?e[i].tensor:e[t];if(o==null)return;let s=this.accumulatedFirstMoment[i].variable,c=this.accumulatedWeightedInfNorm[i].variable,l=G(K(s,this.beta1),K(o,1-this.beta1)),u=K(c,this.beta2),d=Sa(o),f=ko(u,d);s.assign(l),c.assign(f);let p=G(K(ya(r,n),ya(l,G(f,this.epsilon))),a);a.assign(p)}),this.iteration.assign(G(this.iteration,1)),this.accBeta1.assign(K(this.accBeta1,this.beta1))}),this.incrementIterations()}dispose(){this.accBeta1.dispose(),this.iteration.dispose(),this.accumulatedFirstMoment!=null&&Di(this.accumulatedFirstMoment.map(e=>e.variable)),this.accumulatedWeightedInfNorm!=null&&Di(this.accumulatedWeightedInfNorm.map(e=>e.variable))}async getWeights(){throw Error(`getWeights() is not implemented for Adamax yet.`)}async setWeights(e){throw Error(`setWeights() is not implemented for Adamax yet.`)}getConfig(){return{learningRate:this.learningRate,beta1:this.beta1,beta2:this.beta2,epsilon:this.epsilon,decay:this.decay}}static fromConfig(e,t){return new e(t.learningRate,t.beta1,t.beta2,t.epsilon,t.decay)}},ps=class extends cs{static get className(){
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
return`SGD`}constructor(e){super(),this.learningRate=e,this.setLearningRate(e)}applyGradients(e){(Array.isArray(e)?e.map(e=>e.name):Object.keys(e)).forEach((t,n)=>{let r=Array.isArray(e)?e[n].tensor:e[t];if(r==null)return;let i=H.registeredVariables[t];Ei(()=>{let e=G(K(this.c,r),i);i.assign(e)})}),this.incrementIterations()}setLearningRate(e){this.learningRate=e,this.c!=null&&this.c.dispose(),this.c=Oi(vo(-e))}dispose(){this.c.dispose()}async getWeights(){return[await this.saveIterations()]}async setWeights(e){if(e=await this.extractIterations(e),e.length!==0)throw Error(`SGD optimizer does not have settable weights.`)}getConfig(){return{learningRate:this.learningRate}}static fromConfig(e,t){return new e(t.learningRate)}}
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;
/**
* @license
* Copyright 2022 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let ms=[ls,us,ds,fs,class extends ps{static get className(){return`Momentum`}constructor(e,t,n=!1){super(e),this.learningRate=e,this.momentum=t,this.useNesterov=n,this.accumulations=[],this.m=vo(this.momentum)}applyGradients(e){(Array.isArray(e)?e.map(e=>e.name):Object.keys(e)).forEach((t,n)=>{let r=H.registeredVariables[t];this.accumulations[n]??(this.accumulations[n]={originalName:`${t}/momentum`,variable:Ei(()=>io(r).variable(!1))});let i=this.accumulations[n].variable,a=Array.isArray(e)?e[n].tensor:e[t];a!=null&&Ei(()=>{let e,t=G(K(this.m,i),a);e=this.useNesterov?G(K(this.c,G(a,K(t,this.m))),r):G(K(this.c,t),r),i.assign(t),r.assign(e)})}),this.incrementIterations()}dispose(){this.m.dispose(),this.accumulations!=null&&Di(this.accumulations.map(e=>e.variable))}setMomentum(e){this.momentum=e}async getWeights(){return[await this.saveIterations()].concat(this.accumulations.map(e=>({name:e.originalName,tensor:e.variable})))}async setWeights(e){e=await this.extractIterations(e),this.accumulations=e.map(e=>({originalName:e.name,variable:e.tensor.variable(!1)}))}getConfig(){return{learningRate:this.learningRate,momentum:this.momentum,useNesterov:this.useNesterov}}static fromConfig(e,t){return new e(t.learningRate,t.momentum,t.useNesterov)}},class extends cs{static get className(){return`RMSProp`}constructor(e,t=.9,n=0,r=null,i=!1){if(super(),this.learningRate=e,this.decay=t,this.momentum=n,this.epsilon=r,this.accumulatedMeanSquares=[],this.accumulatedMoments=[],this.accumulatedMeanGrads=[],this.centered=i,r??(this.epsilon=H.backend.epsilon()),e==null)throw Error(`learningRate for RMSPropOptimizer must be defined.`)}applyGradients(e){(Array.isArray(e)?e.map(e=>e.name):Object.keys(e)).forEach((t,n)=>{let r=H.registeredVariables[t];this.accumulatedMeanSquares[n]??(this.accumulatedMeanSquares[n]={originalName:`${t}/rms`,variable:Ei(()=>io(r).variable(!1))}),this.accumulatedMoments[n]??(this.accumulatedMoments[n]={originalName:`${t}/momentum`,variable:Ei(()=>io(r).variable(!1))}),this.accumulatedMeanGrads[n]==null&&this.centered&&(this.accumulatedMeanGrads[n]={originalName:`${t}/mg`,variable:Ei(()=>io(r).variable(!1))});let i=Array.isArray(e)?e[n].tensor:e[t];if(i==null)return;let a=this.accumulatedMeanSquares[n].variable,o=this.accumulatedMoments[n].variable;Ei(()=>{let e=G(K(a,this.decay),K(So(i),1-this.decay));if(this.centered){let t=this.accumulatedMeanGrads[n].variable,s=G(K(t,this.decay),K(i,1-this.decay)),c=ya(K(i,this.learningRate),bo(Do(e,G(So(s),this.epsilon)))),l=G(K(o,this.momentum),c);a.assign(e),t.assign(s),o.assign(l);let u=Do(r,l);r.assign(u)}else{let e=G(K(a,this.decay),K(So(i),1-this.decay)),t=G(K(o,this.momentum),ya(K(i,this.learningRate),bo(G(e,this.epsilon))));a.assign(e),o.assign(t);let n=Do(r,t);r.assign(n)}})}),this.incrementIterations()}dispose(){this.accumulatedMeanSquares!=null&&Di(this.accumulatedMeanSquares.map(e=>e.variable)),this.accumulatedMeanGrads!=null&&this.centered&&Di(this.accumulatedMeanGrads.map(e=>e.variable)),this.accumulatedMoments!=null&&Di(this.accumulatedMoments.map(e=>e.variable))}async getWeights(){let e=[...this.accumulatedMeanSquares,...this.accumulatedMoments];return this.centered&&e.push(...this.accumulatedMeanGrads),[await this.saveIterations()].concat(e.map(e=>({name:e.originalName,tensor:e.variable})))}async setWeights(e){e=await this.extractIterations(e);let t=this.centered?e.length/3:e.length/2;this.accumulatedMeanSquares=e.slice(0,t).map(e=>({originalName:e.name,variable:e.tensor.variable(!1)})),this.accumulatedMoments=e.slice(t,t*2).map(e=>({originalName:e.name,variable:e.tensor.variable(!1)})),this.centered&&(this.accumulatedMeanGrads=e.slice(t*2,t*3).map(e=>({originalName:e.name,variable:e.tensor.variable(!1)})))}getConfig(){return{learningRate:this.learningRate,decay:this.decay,momentum:this.momentum,epsilon:this.epsilon,centered:this.centered}}static fromConfig(e,t){return new e(t.learningRate,t.decay,t.momentum,t.epsilon,t.centered)}},ps];function hs(){for(let e of ms)ss(e)}function gs(e,t){let n=e.shape.length,r=t.shape.length;if(n<1)throw Error(`tf.gatherND() expects the input to be rank 1 or higher, but the rank was ${n}.`);if(r<1)throw Error(`tf.gatherND() expects the indices to be rank 1 or higher, but the rank was ${r}.`);if(t.dtype!==`int32`)throw Error(`tf.gatherND() expects the indices to be int32 type, but the dtype was ${t.dtype}.`);if(t.shape[r-1]>n)throw Error(`index innermost dimension length must be <= tensor rank; saw: ${t.shape[r-1]} vs. ${n}`);if(_(e.shape)===0)throw Error(`Requested more than 0 entries, but input is empty. Input shape: ${e.shape}.`);let i=t.shape,a=i[i.length-1],o=1;for(let e=0;e<i.length-1;++e)o*=i[e];let s=e.shape,c=i.slice();c.pop();let l=1;for(let e=a;e<n;++e)l*=s[e],c.push(s[e]);let u=[...P(e.shape).map(e=>e/l),1].slice(0,a);return[c,o,l,u]}
/**
* @license
* Copyright 2021 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function _s(e,t,n){let r=e.shape.length;m(r===t.length,()=>`Error in slice${r}D: Length of begin ${t} must match the rank of the array (${r}).`),m(r===n.length,()=>`Error in slice${r}D: Length of size ${n} must match the rank of the array (${r}).`);for(let i=0;i<r;++i)m(t[i]+n[i]<=e.shape[i],()=>`Error in slice${r}D: begin[${i}] + size[${i}] (${t[i]+n[i]}) would overflow input.shape[${i}] (${e.shape[i]})`)}function vs(e,t,n){let r=[];for(let i=0;i<e.length;i++)r[i]=Math.ceil((t[i]-e[i])/n[i]);return r}function ys(e,t,n){let r=n.length;for(let e=0;e<n.length;e++)if(n[e]>1){r=e;break}for(let i=r+1;i<n.length;i++)if(t[i]>0||n[i]!==e[i])return!1;return!0}function bs(e,t){let n=e.length>0?e[e.length-1]:1;for(let r=0;r<e.length-1;r++)n+=e[r]*t[r];return n}function xs(e,t,n){let r,i=e.shape.length;r=typeof t==`number`?[t,...Array(i-1).fill(0)]:t.length<i?t.concat(Array(i-t.length).fill(0)):t.slice(),r.forEach(e=>{m(e!==-1,()=>`slice() does not support negative begin indexing.`)});let a;return a=n==null?Array(i).fill(-1):typeof n==`number`?[n,...Array(i-1).fill(-1)]:n.length<i?n.concat(Array(i-n.length).fill(-1)):n,a=a.map((t,n)=>t>=0?t:(m(t===-1,()=>`Negative size values should be exactly -1 but got ${t} for the slice() size at index ${n}.`),e.shape[n]-r[n])),[r,a]}function Ss(e,t,n,r,i,a,o,s,c){let l;if(r==null?(l=Array(t.length),l.fill(1)):l=r,o!=null&&o&o-1)throw Error(`Multiple ellipses in slice is not allowed.`);let u=!1,d={dims:l.length,numAddAxisAfterEllipsis:0,begin:t.slice(),end:n.slice(),strides:l.slice(),beginMask:i,endMask:a,ellipsisMask:o,newAxisMask:s,shrinkAxisMask:c};for(let e=0;e<d.dims;e++)u&&1<<e&s&&d.numAddAxisAfterEllipsis++,1<<e&o&&(u=!0);u||(d.ellipsisMask|=1<<d.dims,d.dims++);let f={dims:e.length,beginMask:0,endMask:0,beginValid:!1,endValid:!1};Cs(d,f);let p=!0,m=!0,h=!0,g=[],_=[];for(let t=0;t<e.length;++t){if(f.strides[t]===0)throw Error(`strides[${t}] must be non-zero`);let n=!!(f.shrinkAxisMask&1<<t),r=e[t];if(r===-1){g.push(n?1:-1);continue}let i=[f.beginMask&1<<t,f.endMask&1<<t],a=[f.strides[t]>0?0:-1,f.strides[t]>0?r:r-1];if(n&&f.strides[t]<=0)throw Error(`only stride 1 allowed on non-range indexing.`);h&&=f.strides[t]===1;let o=!!(f.beginMask&1<<t&&f.endMask&1<<t);if(f.beginValid&&f.endValid){if(n){let e=f.begin[t]<0?r+f.begin[t]:f.begin[t];if(f.begin[t]=e,f.end[t]=f.begin[t]+1,e<0||e>=r)throw Error(`slice index ${f.begin[t]} of dimension ${t} out of bounds.`)}else f.begin[t]=ws(f.begin[t],0,f.strides[t],r,i,a),f.end[t]=ws(f.end[t],1,f.strides[t],r,i,a);let e=f.strides[t]===1&&f.begin[t]===0&&f.end[t]===r;p&&=e,m&&=t===0&&f.strides[t]===1||e}else p=p&&f.strides[t]===1&&o,m&&=t===0&&f.strides[t]===1||o;let s,c=!1;if(f.beginValid&&f.endValid?(s=f.end[t]-f.begin[t],c=!0):n?(s=1,c=!0):o&&r>=0&&(s=f.strides[t]<0?-r:r,c=!0),c){let e;e=s===0||s<0!=f.strides[t]<0?0:Math.trunc(s/f.strides[t])+(s%f.strides[t]===0?0:1),g.push(e)}else g.push(-1)}for(let e=0;e<f.finalShapeGatherIndices.length;++e){let t=f.finalShapeGatherIndices[e];t>=0?_.push(g[t]):t===-2&&_.push(1)}return{finalShapeSparse:_.filter((e,t)=>f.finalShapeGatherIndices[t]!==-2),finalShape:_,isIdentity:p,sliceDim0:m,isSimpleSlice:h,begin:f.begin,end:f.end,strides:f.strides}}function Cs(e,t){t.beginMask=0,t.endMask=0,t.shrinkAxisMask=0;let n=0;t.beginValid=e.begin!=null,t.endValid=e.end!=null,t.begin=Array(t.dims),t.end=Array(t.dims),t.strides=Array(t.dims),t.finalShapeGatherIndices=[],t.finalShapeGatherIndicesSparse=[],t.inputShapeGatherIndicesSparse=Array(t.dims);for(let r=0;r<e.dims;r++)if(1<<r&e.ellipsisMask){let i=Math.min(t.dims-(e.dims-r)+1+e.numAddAxisAfterEllipsis,t.dims);for(;n<i;n++)t.begin[n]=0,t.end[n]=0,t.strides[n]=1,t.beginMask|=1<<n,t.endMask|=1<<n,t.finalShapeGatherIndices.push(n),t.finalShapeGatherIndicesSparse.push(-1),t.inputShapeGatherIndicesSparse[n]=r}else if(1<<r&e.newAxisMask)t.finalShapeGatherIndices.push(-2),t.finalShapeGatherIndicesSparse.push(-1);else{if(n===t.begin.length)throw Error(`Index out of range using input dim ${n}; input has only ${t.dims} dims, ${t.begin.length}.`);e.begin!=null&&(t.begin[n]=e.begin[r]),e.end!=null&&(t.end[n]=e.end[r]),t.strides[n]=e.strides[r],e.beginMask&1<<r&&(t.beginMask|=1<<n),e.endMask&1<<r&&(t.endMask|=1<<n),e.shrinkAxisMask&1<<r?(t.finalShapeGatherIndices.push(-1),t.finalShapeGatherIndicesSparse.push(-1),t.shrinkAxisMask|=1<<n):(t.finalShapeGatherIndices.push(n),t.finalShapeGatherIndicesSparse.push(r)),t.inputShapeGatherIndicesSparse[n]=r,n++}}function ws(e,t,n,r,i,a){if(i[t])return n>0?a[t]:a[t+1&1];{let t=e<0?r+e:e;return t<a[0]?a[0]:t>a[1]?a[1]:t}}
/**
* @license
* Copyright 2017 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Ts(e,t){let n=e[0].length;e.forEach((e,t)=>{m(e.length===n,()=>`Error in concat${n}D: rank of tensors[${t}] must be the same as the rank of the rest (${n})`)}),m(t>=0&&t<n,()=>`Error in concat${n}D: axis must be between 0 and ${n-1}.`);let r=e[0];e.forEach((e,i)=>{for(let a=0;a<n;a++)m(a===t||e[a]===r[a],()=>`Error in concat${n}D: Shape of tensors[${i}] (${e}) does not match the shape of the rest (${r}) along the non-concatenated axis ${i}.`)})}function Es(e,t){let n=e[0].slice();for(let r=1;r<e.length;r++)n[t]+=e[r][t];return n}
/**
* @license
* Copyright 2022 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
var Ds;(function(e){e[e.FIRST_DIM_SIZE=0]=`FIRST_DIM_SIZE`,e[e.VALUE_ROWIDS=1]=`VALUE_ROWIDS`,e[e.ROW_LENGTHS=2]=`ROW_LENGTHS`,e[e.ROW_SPLITS=3]=`ROW_SPLITS`,e[e.ROW_LIMITS=4]=`ROW_LIMITS`,e[e.ROW_STARTS=5]=`ROW_STARTS`})(Ds||={});function Os(e,t,n){let r=[];if(n==null&&t==null)return r;if(t==null)for(;r.length<e+n.length;)r.push(-1);else r=t.slice();if(n==null)return r;if(e+n.length!==r.length)throw Error(`rt input.shape and shape=${t} are incompatible: rt input.rank = ${e+n.length}, but shape.rank = ${r.length}`);for(let i=1;i<n.length;++i){let a=n[i],o=r[r.length-n.length+i],s=r[o];if(a>=0){if(s>=0){if(s!==a)throw Error(`rt input.shape and shape=${t} are incompatible: rt input.shape[${i+e}] = ${a} but shape[${i+e}] = ${s}`)}else r[o]=a}}return r}function ks(e){let t={FIRST_DIM_SIZE:Ds.FIRST_DIM_SIZE,VALUE_ROWIDS:Ds.VALUE_ROWIDS,ROW_LENGTHS:Ds.ROW_LENGTHS,ROW_SPLITS:Ds.ROW_SPLITS,ROW_LIMITS:Ds.ROW_LIMITS,ROW_STARTS:Ds.ROW_STARTS},n=[];for(let r of e)if(r in t)n.push(t[r]);else break;return n}function As(e){return e.length===0?0:e[0]===Ds.FIRST_DIM_SIZE?e.length-1:e.length}function js(e,t){if(e==null||t==null)return;let n=e.length,r=t.length;if(n>=r)throw Error(`defaultValue.shape=${e} and ragged tensor flatValues.shape=${t}, are incompatible: defaultValue.rank = ${n} must be less than ragged tensor input flatValues.rank = ${r})`);for(let i=0;i<Math.min(n,r-1);++i){let n=e[i],r=t[i+1];if(n>=0&&r>=0&&n!==1&&n!==r)throw Error(`defaultValue.shape=${e}, and ragged tensor input flatValues.shape=${t} are incompatible: defaultValue.shape[${i-e.length}] = ${n} but ragged tensor input.flatValues.shape[${i-e.length}] = ${r}`)}}
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Ms(e,t,n){return[n*(typeof e==`number`?e:e[0]),t*(typeof e==`number`?e:e[1])]}
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Ns(e,t,n,r=!0){let i=[];if(r)i=i.concat(t.slice(0)),i.push(e[0]/n),i=i.concat(e.slice(1));else{i=i.concat(e[0]);let n=t.length;for(let r=0;r<n;++r)i=i.concat([e[r+1]/t[r],t[r]]);i=i.concat(e.slice(n+1))}return i}function Ps(e,t,n=!0){let r=[];if(n){r.push(t);for(let n=t+1;n<e;++n)n<=2*t?(r.push(n),r.push(n-(t+1))):r.push(n)}else{let n=[],i=[];for(let r=1;r<e;++r)r>=t*2+1||r%2==1?i.push(r):n.push(r);r.push(...n),r.push(0),r.push(...i)}return r}function Fs(e,t,n,r=!0){let i=[];r?i.push(e[0]/n):i.push(e[0]*n);for(let n=1;n<e.length;++n)n<=t.length?r?i.push(t[n-1]*e[n]):i.push(e[n]/t[n-1]):i.push(e[n]);return i}function Is(e,t){let n=[0];for(let r=0;r<t;++r)n.push(e[r][0]);return n}function Ls(e,t,n){let r=e.slice(0,1);for(let i=0;i<n;++i)r.push(e[i+1]-t[i][0]-t[i][1]);return r}
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Rs(e,t){if(e.length!==t.length)throw Error(`Cannot merge real and imag arrays of different lengths. real:${e.length}, imag: ${t.length}.`);let n=new Float32Array(e.length*2);for(let r=0;r<n.length;r+=2)n[r]=e[r/2],n[r+1]=t[r/2];return n}function zs(e){let t=new Float32Array(e.length/2),n=new Float32Array(e.length/2);for(let r=0;r<e.length;r+=2)t[r/2]=e[r],n[r/2]=e[r+1];return{real:t,imag:n}}function Bs(e){let t=Math.ceil(e.length/4),n=new Float32Array(t),r=new Float32Array(t);for(let t=0;t<e.length;t+=4)n[Math.floor(t/4)]=e[t],r[Math.floor(t/4)]=e[t+1];return{real:n,imag:r}}function Vs(e){let t=Math.floor(e.length/4),n=new Float32Array(t),r=new Float32Array(t);for(let t=2;t<e.length;t+=4)n[Math.floor(t/4)]=e[t],r[Math.floor(t/4)]=e[t+1];return{real:n,imag:r}}function Hs(e,t){return{real:e[t*2],imag:e[t*2+1]}}function Us(e,t,n,r){e[r*2]=t,e[r*2+1]=n}function Ws(e,t){let n=new Float32Array(e/2),r=new Float32Array(e/2);for(let i=0;i<Math.ceil(e/2);i++){let a=(t?2:-2)*Math.PI*(i/e);n[i]=Math.cos(a),r[i]=Math.sin(a)}return{real:n,imag:r}}function Gs(e,t,n){let r=(n?2:-2)*Math.PI*(e/t);return{real:Math.cos(r),imag:Math.sin(r)}}
/**
* @license
* Copyright 2021 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let Ks=/->/g;function qs(e,t){e=e.replace(/\s/g,``);let n=(e.length-e.replace(Ks,``).length)/2;if(n<1)throw Error(`Equations without an arrow are not supported.`);if(n>1)throw Error(`Equation must contain exactly one arrow ("->").`);let[r,i]=e.split(`->`);m(r.indexOf(`...`)===-1,()=>`The ellipsis notation ("...") is not supported yet.`);let a=r.split(`,`),o=a.length;if(t!==o)throw Error(`Expected ${o} input tensors, received ${t}`);if(o>2)throw Error(`Support for more than 2 input tensors is not implemented yet.`);let s=[];for(let e=0;e<i.length;++e){let t=i[e];if(!a.some(e=>e.indexOf(t)!==-1))throw Error(`Output subscripts contain the label ${t} not present in the input subscripts.`);s.indexOf(t)===-1&&s.push(t)}for(let e=0;e<r.length;++e){let t=r[e];s.indexOf(t)===-1&&t!==`,`&&s.push(t)}let c=Array(a.length);for(let e=0;e<o;++e){if(new Set(a[e].split(``)).size!==a[e].length)throw Error(`Found duplicate axes in input component ${a[e]}. Support for duplicate axes in input is not implemented yet.`);c[e]=[];for(let t=0;t<a[e].length;++t)c[e].push(s.indexOf(a[e][t]))}let l=s.length,u=i.length,d=[];for(let e=u;e<l;++e)d.push(e);return{allDims:s,summedDims:d,idDims:c}}function Js(e,t){let n=Array(e);n.fill(-1);for(let e=0;e<t.length;++e)n[t[e]]=e;let r=[];for(let t=0;t<e;++t)n[t]===-1&&r.push(t);return n=n.filter(e=>e!==-1),{permutationIndices:n,expandDims:r}}function Ys(e,t,n){let r=Array(e);for(let e=0;e<n.length;++e){let i=n[e].shape;for(let n=0;n<t[e].length;++n)r[t[e][n]]===void 0?r[t[e][n]]=i[n]:m(r[t[e][n]]===i[n],()=>`Expected dimension ${r[t[e][n]]} at axis ${n} of input shaped ${JSON.stringify(i)}, but got dimension ${i[n]}`)}}function Xs(e,t){let n=e,r=[],i=0;e.length===0&&n.push(-1),i=e.length+1;for(let e=0;e<i;++e)r.push([]);let a=[];for(let e=0;e<n.length;++e){let i=n[e],o=Qs(t,i);for(let t of o)a.indexOf(t)===-1&&(r[e].push(t),a.push(t))}return{path:n,steps:r}}function Zs(e){return e.every((e,t)=>e===t)}function Qs(e,t){let n=[];for(let r=0;r<e.length;++r)(e[r].length===0||e[r].indexOf(t)!==-1||t===-1)&&n.push(r);return n}function $s(e,t,n=0){let r=[];if(typeof t==`number`)m(e.shape[n]%t===0,()=>`Number of splits must evenly divide the axis.`),r=Array(t).fill(e.shape[n]/t);else{m(t.reduce((e,t)=>(t===-1&&(e+=1),e),0)<=1,()=>`There should be only one negative value in split array.`);let i=t.indexOf(-1);if(i!==-1){let r=t.reduce((e,t)=>t>0?e+t:e);t[i]=e.shape[n]-r}m(e.shape[n]===t.reduce((e,t)=>e+t),()=>`The sum of sizes must match the size of the axis dimension.`),r=t}return r}
/**
* @license
* Copyright 2021 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function ec(e){return`Received SparseTensor with denseShape[0] = 0 but
  indices.shape[0] = ${e}`}function tc(e,t){return`indices(${e}, 0) is invalid: ${t} < 0`}function nc(e,t,n){return`indices(${e}, 0) is invalid: ${t} >= ${n}`}
/**
* @license
* Copyright 2021 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function rc(e,t){return`only one output dimension may be -1, not both ${e} and ${t}`}function ic(e,t){return`size ${e} must be non-negative, not ${t}`}function ac(){return`reshape cannot infer the missing input size for an empty tensor unless all specified input sizes are non-zero`}function oc(e,t){return`Input to reshape is a SparseTensor with ${_(e)}
  dense values, but the requested shape requires a multiple of ${_(t)}. inputShape=${e} outputShape= ${t}`}function sc(e,t){return`Input to reshape is a tensor with ${_(e)} dense values, but the requested shape has ${_(t)}. inputShape=${e} outputShape=${t}`}
/**
* @license
* Copyright 2021 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function cc(){return`segment ids must be >= 0`}function lc(){return`segment ids are not increasing`}function uc(e,t){return`Segment id ${e} out of range [0, ${t}), possibly because segmentIds input is not sorted.`}function dc(e,t,n){return`Bad: indices[${e}] == ${t} out of range [0, ${n})`}
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function fc(e,t,n,r){let i=t.shape.length,a=e.shape.length;if(r!==0&&(r<-i||r>i))throw Error(`Expect batchDims in the range of [-${i}, ${i}], but got ${r}`);if(r<0&&(r+=i),r>a)throw Error(`batchDims (${r}) must be less than rank(x) (
    ${a}).`);if(n<r)throw Error(`batchDims (${r}) must be less than or equal to axis (${n}).`);for(let n=0;n<r;++n)if(e.shape[n]!==t.shape[n])throw Error(`x.shape[${n}]: ${e.shape[n]} should be equal to indices.shape[${n}]: ${t.shape[n]}.`);let o=e.shape[n],s=[],c=1,l=1,u=1;for(let t=0;t<r;++t)s.push(e.shape[t]),c*=e.shape[t];for(let t=r;t<n;t++)s.push(e.shape[t]),l*=e.shape[t];for(let e=r;e<i;e++)s.push(t.shape[e]);for(let t=n+1;t<a;t++)s.push(e.shape[t]),u*=e.shape[t];return{batchSize:c,sliceSize:u,outerSize:l,dimSize:o,outputShape:s}}
/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function pc(e){try{return e.map(e=>Dr(e))}catch(e){throw Error(`Failed to decode encoded string bytes into utf-8, error: ${e}`)}}function mc(e){return e.map(e=>Er(e))}
/**
* @license
* Copyright 2017 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
hs();
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function q(e,t){Array.isArray(e)||(e=[e]),e.forEach(e=>{e!=null&&m(e.dtype!==`complex64`,()=>`${t} does not support complex64 tensors in the CPU backend.`)})}
/**
* @license
* Copyright 2021 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let hc=Go;var gc=class e extends u{nextDataId(){return e.nextDataId++}constructor(){super(),this.blockSize=48,this.firstUse=!0,this.data=new l(this,Ti())}write(e,t,n){this.firstUse&&(this.firstUse=!1,z().get(`IS_NODE`)&&Yn(`
============================
Hi, looks like you are running TensorFlow.js in Node.js. To speed things up dramatically, install our node backend, visit https://github.com/tensorflow/tfjs-node for more details. 
============================`));let r={id:this.nextDataId()};return this.data.set(r,{values:e,dtype:n,refCount:1}),r}makeTensorInfo(e,t,n){let r;if(t===`string`&&n!=null&&n.length>0&&A(n[0])){let i=n.map(e=>Er(e));r=this.write(i,e,t)}else r=this.write(n,e,t);return{dataId:r,shape:e,dtype:t}}refCount(e){return this.data.has(e)?this.data.get(e).refCount:0}incRef(e){let t=this.data.get(e);t.refCount++}decRef(e){if(this.data.has(e)){let t=this.data.get(e);t.refCount--}}move(e,t,n,r,i){this.data.set(e,{values:t,dtype:r,refCount:i})}numDataIds(){return this.data.numDataIds()}async read(e){return this.readSync(e)}readSync(e){let{dtype:t,complexTensorInfos:n}=this.data.get(e);return t===`complex64`?Rs(this.readSync(n.real.dataId),this.readSync(n.imag.dataId)):ne(this.data.get(e).values,t)}bufferSync(e){let t=this.readSync(e.dataId);if(e.dtype===`string`)try{let n=t.map(e=>Dr(e));return W(e.shape,e.dtype,n)}catch{throw Error(`Failed to decode encoded string bytes into utf-8`)}return W(e.shape,e.dtype,t)}makeOutput(e,t,n){return Ti().makeTensorFromTensorInfo(this.makeTensorInfo(t,n,e),this)}disposeData(e,t=!1){if(this.data.has(e)){if(this.data.get(e).refCount--,!t&&this.data.get(e).refCount>0)return!1;let{complexTensorInfos:n}=this.data.get(e);n!=null&&(this.disposeData(n.real.dataId,!0),this.disposeData(n.imag.dataId,!0)),this.data.delete(e)}return!0}disposeIntermediateTensorInfo(e){this.disposeData(e.dataId)}async time(e){let t=wr();return e(),{kernelMs:wr()-t}}memory(){return{unreliable:!0,reasons:[`The reported memory is an upper bound. Due to automatic garbage collection, the true allocated memory may be less.`]}}where(e){q([e],`where`);let t=this.readSync(e.dataId);return hc(e.shape,t)}dispose(){}floatPrecision(){return 32}epsilon(){return super.epsilon()}};gc.nextDataId=0;
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the License);
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an AS IS BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function _c(e){let t=new Float32Array(e.length);for(let n=0;n<e.length;++n)t[n]=Math.abs(e[n]);return t}let vc={kernelName:`Abs`,backendName:`cpu`,kernelFunc:e=>{let{x:t}=e.inputs,n=e.backend;q(t,`abs`);let r=new Float32Array(_(t.shape)),i=n.data.get(t.dataId).values;return r=_c(i),n.makeOutput(r,t.shape,t.dtype)}};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function yc(e){return(t,n,r,i,a)=>{let o=no(t,n),s=o.length,c=P(o),l=C(a,_(o)),u=t.length,d=n.length,f=P(t),p=P(n),m=to(t,o),h=to(n,o);if(m.length+h.length===0)for(let t=0;t<l.length;++t)l[t]=e(r[t%r.length],i[t%i.length]);else for(let t=0;t<l.length;++t){let n=ae(t,s,c),a=n.slice(-u);m.forEach(e=>a[e]=0);let o=R(a,u,f),g=n.slice(-d);h.forEach(e=>g[e]=0);let _=R(g,d,p);l[t]=e(r[o],i[_])}return[l,o]}}
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function bc(e){let{inputs:t,backend:n}=e,{real:r,imag:i}=t,a=n.data.get(r.dataId).values,o=n.data.get(i.dataId).values,s=n.makeTensorInfo(r.shape,`complex64`),c=n.data.get(s.dataId);return c.complexTensorInfos={real:n.makeTensorInfo(r.shape,`float32`,a),imag:n.makeTensorInfo(i.shape,`float32`,o)},s}let xc={kernelName:`Complex`,backendName:`cpu`,kernelFunc:bc};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Sc(e,t,n=`float32`){if(n===`complex64`)return bc({inputs:{real:Sc(e,t,`float32`),imag:Sc(e,t,`float32`)},backend:e});let r=I(_(t),n);return e.makeTensorInfo(t,n,r)}
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Cc(e){let{inputs:t,backend:n}=e,{x:r}=t;return n.incRef(r.dataId),{dataId:r.dataId,shape:r.shape,dtype:r.dtype}}let wc={kernelName:_t,backendName:`cpu`,kernelFunc:Cc};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Tc(e){let{inputs:t,backend:n}=e,{input:r}=t,i=n.data.get(r.dataId).complexTensorInfos.real,a=n.data.get(i.dataId).values;return n.makeTensorInfo(i.shape,i.dtype,a)}let Ec={kernelName:`Real`,backendName:`cpu`,kernelFunc:Tc};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Dc(e,t,n,r){if(r===`int32`)return[t,`int32`,Int32Array.from(e)];if(r===`bool`){let r=Cr([0],n),[i,a]=yc((e,t)=>e===t?0:1)(t,[],e,r,`bool`);return[a,`bool`,i]}throw Error(`Error in Cast: failed to cast ${n} to ${r}`)}function Oc(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{dtype:a}=r;if(a===`complex64`){if(i.dtype===`complex64`)return Cc({inputs:{x:i},backend:n});let e=Sc(n,i.shape,i.dtype),t=Oc({inputs:{x:i},backend:n,attrs:{dtype:`float32`}}),r=bc({inputs:{real:t,imag:e},backend:n});return n.disposeIntermediateTensorInfo(e),n.disposeIntermediateTensorInfo(t),r}if(i.dtype===`complex64`){let e=Tc({inputs:{input:i},backend:n}),t=Oc({inputs:{x:e},backend:n,attrs:{dtype:a}});return n.disposeIntermediateTensorInfo(e),t}if(!D(i.dtype,a)){let e=Cc({inputs:{x:i},backend:n});return{dataId:e.dataId,shape:e.shape,dtype:a}}let o=n.data.get(i.dataId).values,[s,c,l]=Dc(o,i.shape,i.dtype,a);return n.makeTensorInfo(s,c,l)}let kc={kernelName:Le,backendName:`cpu`,kernelFunc:Oc};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Ac(e,t,n,r){return n==null?({inputs:n,backend:i})=>{let{a,b:o}=n,s=i;q([a,o],e);let c=s.data.get(a.dataId).values,l=s.data.get(o.dataId).values,u=a.dtype===`string`?pc(c):c,d=a.dtype===`string`?pc(l):l,f=r||a.dtype,[p,m]=t(a.shape,o.shape,u,d,f);return s.makeTensorInfo(m,f,p)}:({inputs:e,backend:i})=>{let{a,b:o}=e,s=i;if(a.dtype===`complex64`||o.dtype===`complex64`){let e=Oc({inputs:{x:a},backend:s,attrs:{dtype:`complex64`}}),t=s.data.get(e.dataId),r=t.complexTensorInfos.real,i=t.complexTensorInfos.imag,c=s.data.get(r.dataId).values,l=s.data.get(i.dataId).values,u=Oc({inputs:{x:o},backend:s,attrs:{dtype:`complex64`}}),d=s.data.get(u.dataId),f=d.complexTensorInfos.real,p=d.complexTensorInfos.imag,m=s.data.get(f.dataId).values,h=s.data.get(p.dataId).values,[g,_,v]=n(a.shape,o.shape,c,l,m,h),y=s.makeTensorInfo(v,`float32`,g),b=s.makeTensorInfo(v,`float32`,_),x=bc({inputs:{real:y,imag:b},backend:s});return s.disposeIntermediateTensorInfo(e),s.disposeIntermediateTensorInfo(u),s.disposeIntermediateTensorInfo(y),s.disposeIntermediateTensorInfo(b),x}{let e=s.data.get(a.dataId).values,n=s.data.get(o.dataId).values,i=r||a.dtype,[c,l]=t(a.shape,o.shape,e,n,i);return s.makeTensorInfo(l,i,c)}}}function jc(e){return(t,n,r,i,a,o)=>{let s=no(t,n),c=_(s),l=s.length,u=P(s),d=C(`float32`,c),f=C(`float32`,c),p=to(t,s),m=to(n,s),h=Rs(r,i),g=Rs(a,o),v=t.length,y=P(t),b=n.length,x=P(n);if(p.length+m.length===0)for(let t=0;t<d.length;t++){let n=t%h.length,r=t%g.length,i=e(h[n*2],h[n*2+1],g[r*2],g[r*2+1]);d[t]=i.real,f[t]=i.imag}else for(let t=0;t<d.length;t++){let n=ae(t,l,u),r=n.slice(-v);p.forEach(e=>r[e]=0);let i=R(r,v,y),a=n.slice(-b);m.forEach(e=>a[e]=0);let o=R(a,b,x),s=e(h[i*2],h[i*2+1],g[o*2],g[o*2+1]);d[t]=s.real,f[t]=s.imag}return[d,f,s]}}
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let Mc=Ac(`Add`,yc(((e,t)=>e+t)),jc(((e,t,n,r)=>({real:e+n,imag:t+r})))),Nc={kernelName:`Add`,backendName:`cpu`,kernelFunc:Mc};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Pc(e,t,n,r,i){let a=_(r),o=I(i,n);for(let n=0;n<e.length;n++){let r=e[n];if(r<0)throw Error(`Input x must be non-negative!`);r>=i||(a>0?o[r]+=t[n]:o[r]+=1)}return o}function Fc(e,t,n,r=!1){let i=e.shape[0],a=e.shape[1],o=W([i,n],t.dtype);for(let s=0;s<i;s++)for(let i=0;i<a;i++){let a=e.get(s,i);if(a<0)throw Error(`Input x must be non-negative!`);a>=n||(r?o.set(1,s,a):t.size>0?o.set(o.get(s,a)+t.get(s,i),s,a):o.set(o.get(s,a)+1,s,a))}return o}
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let Ic={kernelName:Fe,backendName:`cpu`,kernelFunc:Ac(Fe,yc(((e,t)=>e&t)))};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Lc(e){return(t,n,r)=>{let i=w(n,t.length);for(let n=0;n<t.length;++n)i[n]=e(t[n],r);return i}}
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function J(e,t,n){return Rc(e,Lc(t),n)}function Rc(e,t,n){return({inputs:r,attrs:i,backend:a})=>{let{x:o}=r;q(o,e);let s=a,c=s.data.get(o.dataId).values,l;if(o.dtype===`string`){if(!Array.isArray(c))throw Error(`String tensor's value was not an instance of Array`);l=pc(c)}else l=c;let u=n||o.dtype,d=t(l,u,i);return s.makeTensorInfo(o.shape,u,d)}}
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the License);
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an AS IS BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let zc={kernelName:Re,backendName:`cpu`,kernelFunc:Rc(Re,Lc(e=>Math.ceil(e)))};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Bc(e,t,n,r){let i=w(n,_(t));if(r&&n!==`string`){let t=0;e.forEach(e=>{let n=_(e.shape);i.set(e.vals,t),t+=n})}else{let r=0;e.forEach(e=>{let a=n===`string`?pc(e.vals):e.vals,o=0;for(let n=0;n<e.shape[0];++n){let s=n*t[1]+r;for(let t=0;t<e.shape[1];++t)i[s+t]=a[o++]}r+=e.shape[1]})}return i}
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let Vc=Ac(at,yc((e,t)=>+(e===t)),null,`bool`),Hc={kernelName:at,backendName:`cpu`,kernelFunc:Vc},Uc=Rc(`Exp`,Lc(e=>Math.exp(e)),`float32`),Wc={kernelName:`Exp`,backendName:`cpu`,kernelFunc:Uc},Gc={kernelName:st,backendName:`cpu`,kernelFunc:Rc(st,Lc(e=>Math.expm1(e)))},Kc={kernelName:ut,backendName:`cpu`,kernelFunc:Rc(ut,Lc(e=>Math.floor(e)))},qc={kernelName:dt,backendName:`cpu`,kernelFunc:Ac(dt,yc((e,t)=>Math.floor(e/t)),null,`int32`)}
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the License);
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an AS IS BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the License);
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an AS IS BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the License);
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an AS IS BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;
/**
* @license
* Copyright 2021 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Jc(e,t,n,r,i,a,o,s,c){let l=W([r,a],n);for(let n=0;n<r;n++){let r=[],u=0;for(let t=0;t<i;t++){let a=e[n*i+t];u+=a*o[t],r.push(a)}if(u<0||u>=c/a)throw Error(`Invalid indices: ${r} does not index into ${s}`);for(let e=0;e<a;e++)l.values[n*a+e]=t.get(...t.indexToLoc(u*a+e))}return l}
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Yc(e,t,n){let r=W(n,e.dtype);for(let n=0;n<r.size;++n){let i=r.indexToLoc(n).slice(),a=i[0],o=i[2],s=t.locToIndex([a,o]);i[2]=t.values[s];let c=e.locToIndex(i);0<=c&&c<e.values.length&&(r.values[n]=e.values[c])}return r}
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let Xc={kernelName:ht,backendName:`cpu`,kernelFunc:Ac(ht,yc((e,t)=>+(e>t)),null,`bool`)},Zc={kernelName:gt,backendName:`cpu`,kernelFunc:Ac(gt,yc((e,t)=>+(e>=t)),null,`bool`)},Qc={kernelName:St,backendName:`cpu`,kernelFunc:Ac(St,yc((e,t)=>+(e<t)),null,`bool`)},$c={kernelName:Ct,backendName:`cpu`,kernelFunc:Ac(Ct,yc((e,t)=>+(e<=t)),null,`bool`)}
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function el(e,t,n){let r=(t-e)/(n-1),i=I(n,`float32`);i[0]=e;for(let e=1;e<i.length;e++)i[e]=i[e-1]+r;return i}
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the License);
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an AS IS BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let tl={kernelName:`Log`,backendName:`cpu`,kernelFunc:Rc(`Log`,Lc(e=>Math.log(e)))};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function nl(e,t,n,r){let i=C(r,_(n));for(let n=0;n<i.length;++n){let r=n*t,a=e[r];for(let n=0;n<t;++n){let t=e[r+n];(Number.isNaN(t)||t>a)&&(a=t)}i[n]=a}return i}
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let rl={kernelName:At,backendName:`cpu`,kernelFunc:Ac(At,yc(((e,t)=>Math.max(e,t))))},il={kernelName:Lt,backendName:`cpu`,kernelFunc:Ac(Lt,yc(((e,t)=>Math.min(e,t))))},al=yc(((e,t)=>e*t)),ol=Ac(Bt,al,jc(((e,t,n,r)=>({real:e*n-t*r,imag:e*r+t*n})))),sl={kernelName:Bt,backendName:`cpu`,kernelFunc:ol}
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;var cl=c(zo());
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function ll(e,t,n){let r=xr(-1,n);return al([],t,r,e,n)}function ul(e){let{inputs:t,backend:n}=e,{x:r}=t;q(r,`neg`);let i=n.data.get(r.dataId).values,[a,o]=ll(i,r.shape,r.dtype);return n.makeTensorInfo(o,r.dtype,a)}let dl={kernelName:`Neg`,backendName:`cpu`,kernelFunc:ul},fl={kernelName:Vt,backendName:`cpu`,kernelFunc:Ac(Vt,yc(((e,t)=>e===t?0:1)),null,`bool`)}
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function pl(e,t,n,r,i){let a=t.length,o=_(t),s=P(t),c=P(i),l=C(n,_(i));for(let t=0;t<o;++t){let n=ae(t,a,s),i=Array(n.length);for(let e=0;e<i.length;e++)i[e]=n[r[e]];let o=R(i,a,c);l[o]=e[t]}return l}
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function ml(e){let{inputs:t,attrs:n,backend:r}=e,{x:i}=t,{perm:a}=n;q(i,`transpose`);let o=i.shape.length,s=Array(o);for(let e=0;e<s.length;e++)s[e]=i.shape[a[e]];let c=r.data.get(i.dataId).values,l=pl(c,i.shape,i.dtype,a,s);return{dataId:r.write(l,s,i.dtype),shape:s,dtype:i.dtype}}let hl={kernelName:Bn,backendName:`cpu`,kernelFunc:ml};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function gl(e,t,n,r){let[i,a]=so(e,r),o=ti(t,`int32`),s=I(_(i),o),c=_(a);for(let e=0;e<s.length;++e){let t=e*c,r=1;for(let e=0;e<c;++e)r*=n[t+e];s[e]=r}return{outVals:s,outShape:i,outDtype:o}}function _l(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{axis:a,keepDims:o}=r;q(i,`prod`);let s=i.shape.length,c=S(a,i.shape),l=uo(c,s),u=c,d=i,f=[];l!=null&&(d=ml({inputs:{x:i},backend:n,attrs:{perm:l}}),f.push(d),u=po(u.length,s));let p=n.data.get(d.dataId).values,{outVals:m,outShape:h,outDtype:g}=gl(d.shape,d.dtype,p,u),_=h;return o&&(_=co(h,c)),f.forEach(e=>n.disposeIntermediateTensorInfo(e)),n.makeTensorInfo(_,g,m)}let vl={kernelName:Xt,backendName:`cpu`,kernelFunc:_l};
/**
* @license
* Copyright 2022 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function yl(e,t,n){e.forEach((e,r)=>{if(e<0||e>=n){let i=ae(r,t.length,P(t)).join(`,`);throw Error(`indices[${i}] = ${e} is not in [0, ${n})`)}})}function bl(e,t){for(let n=0;n<e.length;++n){let r=e[n],i=n===e.length-1?t:e[n+1].length;if(r.length===0)throw Error(`Ragged splits may not be empty`);if(r[0]<0)throw Error(`Ragged splits must be non-negative`);if(r[r.length-1]>i)throw Error(`Ragged splits must not point past values`);for(let e=1;e<r.length;++e)if(r[e-1]>r[e])throw Error(`Ragged splits must be sorted in ascending order`)}}function xl(e,t,n,r){let i=[],a=0,o=t.length-1+n.length,s=Array(o).fill(null).map(()=>[0]);bl(n,r);let c=1;for(let e=0;e<t.length-1;++e){c*=t[e];let n=t[e+1];for(let t=1;t<c+1;++t)s[e].push(t*n)}for(let r=0;r<e.length;++r){let o=e[r],c=e[r]+1;for(let e=0;e<n.length;++e){let r=n[e],i=e+t.length-1;if(i>=0){let e=s[i],t=e[e.length-1]-r[o];for(let e=o;e<c;++e)s[i].push(r[e+1]+t)}o=r[o],c=r[c]}c!==o&&(i.push([o,c]),a+=c-o)}return{outSplits:s,valueSlices:i,numValues:a}}function Sl(e){let t=[];for(let n=0;n<e.length;++n){let r=e[n].length,i=w(`int32`,r);t.push(i),e[n].forEach((e,t)=>i[t]=e)}return t}function Cl(e,t){let n=e.slice(0,t);for(;n.length<t;)n.push(1);for(let r=t;r<e.length;r++)n[t-1]*=e[r];return n}function wl(e,t,n,r,i,a){let o=Cl(t,2)[1],s=Cl(a,2)[1],c=0;for(let t of n)for(let n=t[0];n<t[1];++n){for(let t=0;t<r;++t)i[c*s+t]=e[n*o+t];++c}}function Tl(e,t,n,r,i){let a=t.slice();a[0]=i;let o=w(n,_(a)),s=e.length;return wl(e,t,r,s===0?0:s/t[0],o,a),[o,a]}function El(e,t,n,r,i,a,o,s){if(e.length===0)throw Error(`paramsNestedSplits must be non empty`);if(t[0].length===0)throw Error(`Split tensors must not be scalars`);if(yl(a,o,t[0][0]-1),r.length===0)throw Error(`params.rank must be nonzero`);let c=r[0],{outSplits:l,valueSlices:u,numValues:d}=xl(a,o,e,c),f=Sl(l),p=Tl(n,r,i,u,d);return[f,p[0],p[1]]}
/**
* @license
* Copyright 2022 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let Dl=2147483647;function Ol(e,t,n,r,i,a,o){if(t.length>1)throw Error(`starts must be a scalar or vector`);if(i.length>1)throw Error(`limits must be a scalar or vector`);if(o.length>1)throw Error(`deltas must be a scalar or vector`);let s=t.length===0,c=i.length===0,l=o.length===0,u=[];s||u.push(t[0]),c||u.push(i[0]),l||u.push(o[0]);for(let e=1;e<u.length;++e)if(u[e]!==u[e-1])throw Error(`starts, limits, and deltas must have the same shape`);let d=u.length===0?1:u[0],f=w(`int32`,d+1);f[0]=0;for(let t=0;t<d;++t){let n=s?e[0]:e[t],i=c?r[0]:r[t],o=l?a[0]:a[t];if(o===0)throw Error(`Requires delta != 0`);let u;if(o>0&&i<n||o<0&&i>n)u=0;else if(u=Math.ceil(Math.abs((i-n)/o)),u>Dl)throw Error(`Requires ((limit - start) / delta) <= ${Dl}`);f[t+1]=f[t]+u}let p=f[d],m=w(n,p),h=0;for(let t=0;t<d;++t){let n=f[t+1]-f[t],r=s?e[0]:e[t],i=l?a[0]:a[t];for(let e=0;e<n;++e)m[h++]=r,r+=i}return[f,m]}
/**
* @license
* Copyright 2022 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
var kl=Ds,Al=class e{constructor(e,t,n,r,i,a,o,s,c,l){this.shape=e,this.shapeShape=t,this.values=n,this.valuesShape=r,this.valuesDType=i,this.defaultValue=a,this.defaultValueShape=o,this.rowPartitionValues=s,this.rowPartitionValuesShapes=c,this.rowPartitionTypes=ks(l),this.raggedRank=As(this.rowPartitionTypes)}getRowPartitionTypeByDimension(e){return this.rowPartitionTypes[0]===kl.FIRST_DIM_SIZE?this.rowPartitionTypes[e+1]:this.rowPartitionTypes[e]}getRowPartitionTensor(e){return this.rowPartitionTypes[0]===kl.FIRST_DIM_SIZE?this.rowPartitionValues[e+1]:this.rowPartitionValues[e]}getMaxWidth(t){let n=this.getRowPartitionTensor(t-1);switch(this.getRowPartitionTypeByDimension(t-1)){case kl.VALUE_ROWIDS:return e.getMaxWidthValueRowID(n);case kl.ROW_SPLITS:return e.getMaxWidthRowSplit(n);default:throw Error(`Cannot handle partition type ${kl[this.getRowPartitionTypeByDimension(t-1)]}`)}}static getMaxWidthRowSplit(e){let t=e.length;if(t===0||t===1)return 0;let n=0;for(let r=0;r<t-1;++r){let t=e[r+1]-e[r];t>n&&(n=t)}return n}static getMaxWidthValueRowID(e){let t=e.length;if(t===0)return 0;let n=0,r=e[0],i=0;for(let a=1;a<t;++a){let t=e[a];t!==r&&(r=t,i=Math.max(a-n,i),n=a)}return Math.max(t-n,i)}tensorShapeFromTensor(e,t,n=!0){if(t.length===0){if(e[0]===-1)return[];throw Error(`The only valid scalar shape tensor is the fully unknown shape specified as -1.`)}return Ml(e,n)}calculateOutputSize(e){let t=this.valuesShape,n=this.defaultValueShape;js(n,t);let r=this.tensorShapeFromTensor(this.shape,this.shapeShape),i=Os(this.raggedRank,r,t);i[0]<0&&(i[0]=e);for(let e=1;e<=this.raggedRank;++e)i[e]<0&&(i[e]=this.getMaxWidth(e));return i}calculateFirstParentOutputIndex(e,t,n){let r=Math.min(e,n),i=[],a=0;for(let e=0;e<r;++e,a+=t)i.push(a);for(let t=r;t<e;++t)i.push(-1);return m(i.length===e,()=>`Final length of result must be equal to firstDimension.`),i}calculateOutputIndexRowSplit(e,t,n,r){let i=e.length,a=[];for(let o=0;o<i-1;++o){let i=e[o+1]-e[o],s=Math.min(r,i),c=t[o];c===-1&&(s=0);for(let e=0;e<s;++e)a.push(c),c+=n;for(let e=0;e<i-s;++e)a.push(-1)}if(i>0&&a.length!==e[i-1])throw Error(`Invalid row split size.`);return a}calculateOutputIndexValueRowID(e,t,n,r){let i=e.length,a=[];if(i===0)return[];let o=0,s=e[0];if(s>=t.length)throw Error(`Got currentValueRowId=${s}, which is not less than ${t.length}`);let c=t[s];a.push(c);for(let l=1;l<i;++l){let i=e[l];if(i===s)c>=0&&(++o,o<r?c+=n:c=-1);else{if(o=0,s=i,i>=t.length)throw Error(`Got nextValueRowId=${i} which is not less than ${t.length}`);c=t[i]}a.push(c)}if(a.length!==e.length)throw Error(`Invalid row ids.`);return a}calculateOutputIndex(e,t,n,r){let i=this.getRowPartitionTensor(e),a=this.getRowPartitionTypeByDimension(e);switch(a){case kl.VALUE_ROWIDS:return this.calculateOutputIndexValueRowID(i,t,n,r);case kl.ROW_SPLITS:if(i.length-1>t.length)throw Error(`Row partition size is greater than output size: ${i.length-1} > ${t.length}`);return this.calculateOutputIndexRowSplit(i,t,n,r);default:throw Error(`Unsupported partition type: ${kl[a]}`)}}getFirstDimensionSize(){let e=this.rowPartitionValues[0];if(this.rowPartitionTypes.length===0)throw Error(`No row_partition_types given.`);let t=this.rowPartitionTypes[0];switch(t){case kl.FIRST_DIM_SIZE:return e[0];case kl.VALUE_ROWIDS:throw Error(`Cannot handle VALUE_ROWIDS in first dimension.`);case kl.ROW_SPLITS:return this.rowPartitionValuesShapes[0][0]-1;default:throw Error(`Cannot handle type ${kl[t]}`)}}compute(){if(this.rowPartitionValues[0].length<=0)throw Error(`Invalid first partition input. Tensor requires at least one element.`);let e=this.getFirstDimensionSize(),t=this.calculateOutputSize(e),n=Array(this.raggedRank+1);n[n.length-1]=1;for(let e=n.length-2;e>=0;--e)n[e]=n[e+1]*t[e+1];let r=Ml(t,!1),i=w(this.valuesDType,_(r));if(n[0]*t[0]>0){let a=this.calculateFirstParentOutputIndex(e,n[0],t[0]);for(let e=1;e<=this.raggedRank;++e)a=this.calculateOutputIndex(e-1,a,n[e],t[e]);this.setOutput(this.raggedRank,a,i,r)}return[r,i]}setOutput(e,t,n,r){if(n.length===0)return;let i=this.values,a=n,o=r.slice();o=o.slice(e+1);let s=_(o),c=t.length,l=this.defaultValue;if(l.length!==s&&l.length!==1){let e=this.defaultValueShape;Ei(()=>{let t=Ua(l,e);l=Za(t,o).dataSync()})}let u=0,d=0,f=0;for(let e=0;e<=c;++e){let r=e<c?t[e]:-1;if(r===f){++f;continue}if(d<f){let e=i.subarray(u*s);jl(a.subarray(d*s),e,(f-d)*s)}if(e>=c){let e=n.length;r=Math.floor(e/s)}if(r>f){if(this.defaultValue.length===1)a.subarray(f*s,r*s).fill(this.defaultValue[0]),f=r;else for(;r>f;)jl(a.slice(f*s),l,s),++f}r<0?(u=e+1,d=f):(u=e,d=f,f=d+1)}}};function jl(e,t,n){for(let r=0;r<n;r++)e[r]=t[r]}function Ml(e,t){let n=[];for(let r of e){if(r<0){if(!t)throw Error(`Dimension ${r} must be >= 0`);if(r<-1)throw Error(`Dimension ${r} must be >= -1`);r=-1}n.push(r)}return n}function Nl(e,t,n,r,i,a,o,s,c,l){return new Al(e,t,n,r,i,a,o,s,c,l).compute()}
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Pl(e,t,n,r){if(e===t||e<t&&n<0||t<e&&n>1)return I(0,r);let i=I(Math.abs(Math.ceil((t-e)/n)),r);t<e&&n===1&&(n=-1),i[0]=e;for(let e=1;e<i.length;e++)i[e]=i[e-1]+n;return i}
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the License);
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an AS IS BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let Fl={kernelName:ln,backendName:`cpu`,kernelFunc:Rc(ln,Lc(e=>1/Math.sqrt(e)))};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Il(e,t,n,r,i,a,o,s,c,l){let u=[r/i,i],d=e.values,f=t.values;if(r===0)return W(n,t.dtype);let p=c instanceof Vr?c:W(u,t.dtype);typeof c==`string`||typeof c==`number`?p.values.fill(c):typeof c==`boolean`&&p.values.fill(+c);for(let e=0;e<a;e++){let a=[],c=0;for(let t=0;t<o;t++){let n=d[e*o+t];a.push(n),c+=n*s[t]}if(c<0||c>=r/i)throw Error(`Invalid indices: ${a} does not index into ${n}`);for(let n=0;n<i;n++)l?p.values[c*i+n]+=f[e*i+n]:p.values[c*i+n]=t.rank===0?f[0]:f[e*i+n]}return p}
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the License);
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an AS IS BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let Ll=J(vn,e=>1/(1+Math.exp(-e))),Rl={kernelName:vn,backendName:`cpu`,kernelFunc:Ll};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function zl(e,t,n,r,i){let a=ys(r,t,n),o=_(n),s=P(r);if(a){let n=bs(t,s);return i===`string`?e.slice(n,n+o):e.subarray(n,n+o)}let c=W(r,i,i===`string`?pc(e):e),l=W(n,i);for(let e=0;e<l.size;++e){let n=l.indexToLoc(e),r=n.map((e,n)=>e+t[n]);l.set(c.get(...r),...n)}return i===`string`?mc(l.values):l.values}function Bl(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{begin:a,size:o}=r;q(i,`slice`);let[s,c]=xs(i,a,o);_s(i,s,c);let l=n.data.get(i.dataId).values,u=zl(l,s,c,i.shape,i.dtype);return n.makeTensorInfo(c,i.dtype,u)}let Vl={kernelName:hn,backendName:`cpu`,kernelFunc:Bl};
/**
* @license
* Copyright 2021 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Hl(e,t,n,r,i,a,o){let s=t[0],c=a[0],l=Array(c),u=Array(s),d=t[1];if(c===0){if(s!==0)throw Error(ec(s));let e=w(n,0),t=w(i,0);return[e,[0,d],t,l,u]}let f=!0,p=0,m=Array(c).fill(0);for(let t=0;t<s;++t){let n=e[t*d];if(n<0)throw Error(tc(t,n));if(n>=c)throw Error(nc(t,n,c));++m[n],f&&=n>=p,p=n}let h=!0;for(let e=0;e<c;++e){let t=m[e]===0;l[e]=t,h&&=!t,m[e]=Math.max(m[e],1),e>0&&(m[e]+=m[e-1])}if(h&&f){let t=e,n=r;for(let e=0;e<s;++e)u[e]=e;return[t,[s,d],n,l,u]}{let t=m[c-1],a=w(n,t*d),f=w(i,t),p=Array(c).fill(0);for(let t=0;t<s;++t){let n=e[t*d],i=p[n],o=(n===0?0:m[n-1])+i;p[n]++;for(let n=0;n<d;++n)a[o*d+n]=e[t*d+n];f[o]=r[t],u[t]=o}for(let e=0;e<c;++e)if(p[e]===0){let t=e===0?0:m[e-1];a[t*d+0]=e;for(let e=1;e<d;++e)a[t*d+e]=0;f[t]=o}return[a,[t,d],f,l,u]}}
/**
* @license
* Copyright 2021 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Ul(e,t,n,r,i){let a=_(r),o=t[0],s=i.length,c=[],l=1,u=-1;for(let e=0;e<s;++e){let t=i[e];if(t===-1){if(u!==-1)throw Error(rc(u,e));u=e,c.push(1)}else{if(t<0)throw Error(ic(e,t));l*=t,c.push(t)}}if(u!==-1){if(l<=0)throw Error(ac());let e=Math.trunc(a/l);if(l*e!==a)throw Error(oc(r,c));c[u]=e}if(_(c)!==a)throw Error(sc(r,c));let d=r.length,f=[];if(d>0){f[d-1]=1;for(let e=d-2;e>=0;--e)f[e]=f[e+1]*r[e+1]}let p=[];if(s>0){p[s-1]=1;for(let e=s-2;e>=0;--e)p[e]=p[e+1]*c[e+1]}let m=w(n,o*s);for(let t=0;t<o;++t){let n=0;for(let r=0;r<d;++r)n+=e[t*d+r]*f[r];for(let e=0;e<s;++e)m[t*s+e]=Math.trunc(n/p[e]),n%=p[e]}return[m,[o,s],c]}
/**
* @license
* Copyright 2021 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Wl(e,t,n,r,i,a=!1,o=0){let s=r.length,c=[t[0],e.length/t[0]],l=c[1],u=s>0?i[s-1]+1:0;if(u<0)throw Error(cc());let d=t.slice();d[0]=u;let f=w(n,d.reduce((e,t)=>e*t,1));if(s===0)return u>0&&f.fill(o),[f,d];if(u<=0)throw Error(cc());let p=0,m=1,h=0,g=i[p];for(;;){let t=0;if(m<s){if(t=i[m],g===t){++m;continue}if(g>=t)throw Error(lc())}if(g<0||g>=u)throw Error(uc(g,u));g>h&&f.fill(o,h*l,g*l);for(let t=p;t<m;++t){let n=r[t];if(n<0||n>=c[0])throw Error(dc(t,r[t],c[0]));for(let t=0;t<l;t++)f[g*l+t]+=e[n*l+t]}if(a)for(let e=0;e<l;e++)f[g*l+e]/=m-p;if(p=m,++m,h=g+1,g=t,m>s)break}return h<u&&f.fill(o,h*l,u*l),[f,d]}
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the License);
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an AS IS BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let Gl={kernelName:bn,backendName:`cpu`,kernelFunc:J(bn,e=>Math.sqrt(e))},Kl={kernelName:kn,backendName:`cpu`,kernelFunc:Ac(kn,yc(((e,t)=>{
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let n=e-t;return n*n})))},ql={kernelName:jn,backendName:`cpu`,kernelFunc:Rc(jn,Lc((e,t)=>{
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let{pattern:n,replaceGlobal:r,rewrite:i}=t;return e.replace(new RegExp(n,r?`g`:``),i)}))};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Jl(e,t,n,r){let i=W(e,t.dtype);for(let e=0;e<i.size;e++){let a=i.indexToLoc(e),o=Array(a.length);for(let e=0;e<o.length;e++)o[e]=a[e]*n[e]+r[e];i.set(t.get(...o),...a)}return i}
/**
* @license
* Copyright 2021 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
var Yl=class{constructor(e,t,n,r,i,a){this.separator=Er(e),this.nGramWidths=t,this.leftPad=Er(n),this.rightPad=Er(r),this.padWidth=i,this.preserveShort=a}getPadWidth(e){return Math.min(this.padWidth<0?e-1:this.padWidth,e-1)}getNumNGrams(e,t){let n=this.getPadWidth(t);return Math.max(0,e+2*n-t+1)}createNGrams(e,t,n,r,i,a){for(let o=0;o<i;++o){let s=this.getPadWidth(a),c=Math.max(0,s-o),l=Math.max(0,s-(i-(o+1))),u=a-(c+l),d=t+(c>0?0:o-s),f=0;f+=c*this.leftPad.length;for(let t=0;t<u;++t)f+=e[d+t].length;f+=l*this.rightPad.length;let p=c+l+u-1;f+=p*this.separator.length,n[r+o]=new Uint8Array(f);let m=n[r+o],h=0,g=e=>e.forEach(e=>m[h++]=e);for(let e=0;e<c;++e)g(this.leftPad),g(this.separator);for(let t=0;t<u-1;++t)g(e[d+t]),g(this.separator);if(u>0){g(e[d+u-1]);for(let e=0;e<l;++e)g(this.separator),g(this.rightPad)}else{for(let e=0;e<l-1;++e)g(this.rightPad),g(this.separator);g(this.rightPad)}}}compute(e,t){let n=e.length,r=t.length;if(r>0){let e=t[0];if(e!==0)throw Error(`First split value must be 0, got ${e}`);for(let i=1;i<r;++i){let r=t[i]>=e;if(r&&=t[i]<=n,!r)throw Error(`Invalid split value ${t[i]}, must be in [${e}, ${n}]`);e=t[i]}if(e!==n)throw Error(`Last split value must be data size. Expected ${n}, got ${e}`)}let i=r-1,a=w(`int32`,r);if(n===0||r===0){let e=Array(n);for(let e=0;e<=i;++e)a[e]=0;return[e,a]}a[0]=0;for(let e=1;e<=i;++e){let n=t[e]-t[e-1],r=0;this.nGramWidths.forEach(e=>{r+=this.getNumNGrams(n,e)}),this.preserveShort&&n>0&&r===0&&(r=1),a[e]=a[e-1]+r}let o=Array(a[i]);for(let n=0;n<i;++n){let r=t[n],i=a[n];if(this.nGramWidths.forEach(a=>{let s=t[n+1]-t[n],c=this.getNumNGrams(s,a);this.createNGrams(e,r,o,i,c,a),i+=c}),this.preserveShort&&i===a[n]){let a=t[n+1]-t[n];if(a===0)continue;let s=a+2*this.padWidth;this.createNGrams(e,r,o,i,1,s)}}return[o,a]}};function Xl(e,t,n,r,i,a,o,s){return new Yl(n,r,i,a,o,s).compute(e,t)}
/**
* @license
* Copyright 2021 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Zl(e,t,n,r){if(!e.length)return;if(t.length===0){for(let t=0;t<e.length;++t)r.push(e.subarray(t,t+1));return}if(t.length===1){let i=t[0],a=e.indexOf(i);for(;a!==-1;){let t=e.subarray(0,a);(!n||t.length!==0)&&r.push(t),e=e.subarray(a+1),a=e.indexOf(i)}(!n||e.length!==0)&&r.push(e);return}let i=0;for(let a=0;a<e.length+1;a++)if(a===e.length||t.indexOf(e[a])!==-1){let t=e.subarray(i,a);(!n||t.length!==0)&&r.push(t),i=a+1}}function Ql(e,t,n){let r=e.length,i=[],a=0,o=0,s=Array(r);for(let c=0;c<r;++c){let r=i.length;Zl(e[c],t,n,i);let l=i.length-r;s[c]=l,a+=l,o=Math.max(o,l)}let c=w(`int32`,a*2),l=Array(a),u=[r,o],d=0;for(let e=0;e<r;++e)for(let t=0;t<s[e];++t)c[d*2]=e,c[d*2+1]=t,l[d]=i[d],++d;return[c,l,u]}
/**
* @license
* Copyright 2021 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function $l(e,t){let n=w(`int32`,e.length);for(let r=0;r<e.length;++r)n[r]=br(e[r]).modulo(t).getLowBitsUnsigned();return n}
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let eu=Ac(`Sub`,yc(((e,t)=>e-t)),jc(((e,t,n,r)=>({real:e-n,imag:t-r})))),tu={kernelName:`Sub`,backendName:`cpu`,kernelFunc:eu};
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function nu(e,t){let n=Array(e.rank);for(let r=0;r<n.length;r++)n[r]=e.shape[r]*t[r];let r=W(n,e.dtype);for(let t=0;t<r.values.length;++t){let n=r.indexToLoc(t),i=Array(e.rank);for(let t=0;t<i.length;t++)i[t]=n[t]%e.shape[t];let a=e.locToIndex(i);r.values[t]=e.values[a]}return r}
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let ru=(e,t)=>{let n=t.value-e.value;return n===0?e.index-t.index:n};function iu(e,t,n=0,r=e.length-1){for(;r>n;){if(r-n>600){let i=r-n+1,a=t-n+1,o=Math.log(i),s=.5*Math.exp(2*o/3),c=.5*Math.sqrt(o*s*(i-s)/i)*Math.sign(a-i/2);iu(e,t,Math.max(n,Math.floor(t-a*s/i+c)),Math.min(r,Math.floor(t+(i-a)*s/i+c)))}let i=e[t],a=n,o=r;for(p(e,n,t),ru(e[r],i)>0&&p(e,n,r);a<o;){for(p(e,a,o),a++,o--;ru(e[a],i)<0;)a+=1;for(;ru(e[o],i)>0;)--o}ru(e[n],i)===0?p(e,n,o):(o+=1,p(e,o,r)),o<=t&&(n=o+1),t<=o&&(r=o-1)}}function au(e,t,n,r,i){let a=t[t.length-1],[o,s]=[e.length/a,a],c=C(n,o*r),l=C(`int32`,o*r);for(let t=0;t<o;t++){let n=t*s,a=e.subarray(n,n+s),o=Array(a.length);a.forEach((e,t)=>o[t]={value:e,index:t}),r<o.length&&(iu(o,r),o=o.slice(0,r)),i&&o.sort(ru);let u=t*r,d=c.subarray(u,u+r),f=l.subarray(u,u+r);for(let e=0;e<r;e++)d[e]=o[e].value,f[e]=o[e].index}let u=t.slice();return u[u.length-1]=r,[W(u,n,c),W(u,`int32`,l)]}
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function ou(e,t,n,r){let i=S(t,n)[0],a=[1,n[0],1];for(let e=0;e<i;e++)a[0]*=n[e];a[1]=n[i];for(let e=i+1;e<n.length;e++)a[2]*=n[e];let o=new Map,s=new Int32Array(n[i]),c=new Vr(a,r,e),l=[],u=a[0]===1&&a[2]===1;for(let t=0;t<n[i];t++){let n;if(u)n=e[t].toString();else{let e=[];for(let n=0;n<a[0];n++)for(let r=0;r<a[2];r++)e.push(c.get(n,t,r));n=e.join(`,`)}let r=o.get(n);if(r!=null)s[t]=r;else{let e=o.size;o.set(n,e),s[t]=e,l.push(t)}}let d=a.slice();d[1]=o.size;let f=new Vr(d,r);l.forEach((e,t)=>{for(let n=0;n<a[0];n++)for(let r=0;r<a[2];r++)f.set(c.get(n,e,r),n,t,r)});let p=n.slice();return p[i]=d[1],{outputValues:f.values,outputShape:p,indices:s}}
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
Mi(`cpu`,()=>new gc,1);
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the License);
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an AS IS BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let su=J(`Elu`,e=>e>=0?e:Math.exp(e)-1),cu={kernelName:`Elu`,backendName:`cpu`,kernelFunc:su};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function lu(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{alpha:a}=r;q([i],`leakyRelu`);let o=_(i.shape),s=n.data.get(i.dataId).values,c=C(`float32`,o);for(let e=0;e<s.length;e++)c[e]=s[e]<0?a*s[e]:s[e];return n.makeTensorInfo(i.shape,`float32`,c)}let uu={kernelName:xt,backendName:`cpu`,kernelFunc:lu},du=yc((e,t)=>e<0?t*e:e)
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the License);
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an AS IS BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;function fu(e){let{inputs:t,backend:n}=e,{x:r,alpha:i}=t;q([r,i],`prelu`);let a=n.data.get(r.dataId).values,o=n.data.get(i.dataId).values,[s,c]=du(r.shape,i.shape,a,o,`float32`);return n.makeTensorInfo(c,`float32`,s)}let pu={kernelName:Yt,backendName:`cpu`,kernelFunc:fu},mu=J($t,e=>Math.max(0,e)),hu={kernelName:$t,backendName:`cpu`,kernelFunc:mu},gu=J(on,e=>Math.min(Math.max(0,e),6)),_u={kernelName:on,backendName:`cpu`,kernelFunc:gu}
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the License);
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an AS IS BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the License);
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an AS IS BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function vu(e,t,n,r,i){if(n===`linear`)return Cc({inputs:{x:t},backend:e});if(n===`relu`)return mu({inputs:{x:t},backend:e});if(n===`elu`)return su({inputs:{x:t},backend:e});if(n===`relu6`)return gu({inputs:{x:t},backend:e});if(n===`prelu`)return fu({inputs:{x:t,alpha:r},backend:e});if(n===`leakyrelu`)return lu({inputs:{x:t},backend:e,attrs:{alpha:i}});if(n===`sigmoid`)return Ll({inputs:{x:t},backend:e});throw Error(`Activation ${n} has not been implemented for the CPU backend.`)}
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function yu(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{shape:a}=r,o=_(i.shape),s=x(a,o),c=_(s);m(o===c,()=>`The new shape (${s}) has ${c} elements and the old shape (${i.shape}) has ${o} elements. The new shape and old shape must have the same number of elements.`),n.incRef(i.dataId);let l=n.data.get(i.dataId);if(l.complexTensorInfos!=null){let e=l.complexTensorInfos.real,t=l.complexTensorInfos.imag;e.shape=s,t.shape=s}return{dataId:i.dataId,shape:s,dtype:i.dtype}}let bu={kernelName:en,backendName:`cpu`,kernelFunc:yu};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the License);
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an AS IS BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function xu(e){let{inputs:t,backend:n,attrs:r}=e,{a:i,b:a}=t,{transposeA:o,transposeB:s}=r;q([i,a],`matMul`);let c=i.shape.length,l=a.shape.length,u=o?i.shape[c-2]:i.shape[c-1],d=s?a.shape[l-1]:a.shape[l-2],f=o?i.shape[c-1]:i.shape[c-2],p=s?a.shape[l-2]:a.shape[l-1],h=i.shape.slice(0,-2),g=a.shape.slice(0,-2),v=_(h),y=_(g),b=no(i.shape.slice(0,-2),a.shape.slice(0,-2)).concat([f,p]);m(u===d,()=>`Error in matMul: inner shapes (${u}) and (${d}) of Tensors with shapes ${i.shape} and ${a.shape} and transposeA=${o} and transposeB=${s} must match.`);let x=o?[v,u,f]:[v,f,u],S=s?[y,p,d]:[y,d,p],C=yu({inputs:{x:i},backend:n,attrs:{shape:x}}),w=yu({inputs:{x:a},backend:n,attrs:{shape:S}}),T=o?C.shape[1]:C.shape[2],E=o?C.shape[2]:C.shape[1],D=s?w.shape[1]:w.shape[2],O=Math.max(v,y),k=n.data.get(C.dataId).values,A=n.data.get(w.dataId).values,j=P(C.shape),M=P(w.shape),[N,ee,te]=o?[j[0],1,j[1]]:[j[0],j[1],1],[F,ne,re]=s?[1,M[1],M[0]]:[M[1],1,M[0]],I=E*D,ie=W([O,E,D],C.dtype),L=ie.values,R=n.blockSize;for(let e=0;e<O;e++){let t=e%v,n=e%y;for(let r=0;r<E;r+=R){let i=Math.min(r+R,E);for(let a=0;a<D;a+=R){let o=Math.min(a+R,D);for(let s=0;s<T;s+=R){let c=Math.min(s+R,T);for(let l=r;l<i;l++)for(let r=a;r<o;r++){let i=0;for(let e=s;e<c;e++){let a=k[t*N+l*ee+e*te],o=A[e*F+r*ne+n*re];i+=a*o}L[e*I+(l*D+r)]+=i}}}}}return n.disposeIntermediateTensorInfo(C),n.disposeIntermediateTensorInfo(w),n.makeTensorInfo(b,ie.dtype,ie.values)}let Su={kernelName:Me,backendName:`cpu`,kernelFunc:xu};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the License);
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an AS IS BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Cu(e){let{inputs:t,backend:n,attrs:r}=e,{a:i,b:a,bias:o,preluActivationWeights:s}=t,{transposeA:c,transposeB:l,activation:u,leakyreluAlpha:d}=r,f,p,m,h=[];f=xu({inputs:{a:i,b:a},attrs:{transposeA:c,transposeB:l},backend:n}),o&&(p=Mc({inputs:{a:f,b:o},backend:n}),h.push(f),f=p),u&&(m=vu(n,f,u,s,d),h.push(f),f=m);for(let e of h)n.disposeIntermediateTensorInfo(e);return f}let wu={kernelName:Kn,backendName:`cpu`,kernelFunc:Cu},Tu={kernelName:ve,backendName:`cpu`,kernelFunc:J(ve,e=>Math.acos(e))},Eu={kernelName:ye,backendName:`cpu`,kernelFunc:J(ye,e=>Math.acosh(e))}
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the License);
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an AS IS BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the License);
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an AS IS BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Du(e){let{inputs:t,backend:n}=e,r=t;q(t,`addN`);let i=r.map(e=>n.data.get(e.dataId).values),a=W(r[0].shape,r[0].dtype),o=a.values;for(let e=0;e<r.length;e++){let t=i[e];for(let e=0;e<o.length;e++)o[e]+=t[e]}return n.makeTensorInfo(a.shape,a.dtype,a.values)}let Ou={kernelName:be,backendName:`cpu`,kernelFunc:Du};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function ku(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{axis:a,keepDims:o}=r;q(i,`all`);let s=S(a,i.shape),c=s,l=uo(c,i.shape.length),u=i;l!=null&&(u=ml({inputs:{x:i},backend:n,attrs:{perm:l}}),c=po(c.length,i.shape.length)),lo(`all`,c,u.shape.length);let[d,f]=so(u.shape,c),p=_(f),m=I(_(d),u.dtype),h=n.data.get(u.dataId).values;for(let e=0;e<m.length;++e){let t=e*p,n=h[t];for(let e=0;e<p;++e){let r=h[t+e];n&&=r}m[e]=n}l!=null&&n.disposeIntermediateTensorInfo(u);let g=n.makeTensorInfo(d,u.dtype,m);if(o){let e=co(d,s),t=yu({inputs:{x:g},backend:n,attrs:{shape:e}});return n.disposeIntermediateTensorInfo(g),t}return g}let Au={kernelName:`All`,backendName:`cpu`,kernelFunc:ku};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function ju(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{axis:a,keepDims:o}=r;q(i,`any`);let s=S(a,i.shape),c=s,l=uo(c,i.shape.length),u=i;l!=null&&(u=ml({inputs:{x:i},backend:n,attrs:{perm:l}}),c=po(c.length,i.shape.length)),lo(`any`,c,u.shape.length);let[d,f]=so(u.shape,c),p=_(f),m=I(_(d),u.dtype),h=n.data.get(u.dataId).values;for(let e=0;e<m.length;++e){let t=e*p,n=h[t];for(let e=0;e<p;++e){let r=h[t+e];n||=r}m[e]=n}l!=null&&n.disposeIntermediateTensorInfo(u);let g=n.makeTensorInfo(d,u.dtype,m);if(o){let e=co(d,s),t=yu({inputs:{x:g},backend:n,attrs:{shape:e}});return n.disposeIntermediateTensorInfo(g),t}return g}let Mu={kernelName:`Any`,backendName:`cpu`,kernelFunc:ju};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Nu(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{axis:a}=r;q(i,`argMax`);let o=S(a,i.shape),s=uo(o,i.shape.length),c=i,l=[];s!=null&&(c=ml({inputs:{x:i},backend:n,attrs:{perm:s}}),l.push(c),o=po(o.length,c.shape.length)),o=[o[0]],lo(`argMax`,o,c.shape.length);let[u,d]=so(c.shape,o),f=I(_(u),`int32`),p=_(d),m=n.data.get(c.dataId).values;for(let e=0;e<f.length;++e){let t=e*p,n=m[t],r=0;for(let e=0;e<p;++e){let i=m[t+e];i>n&&(n=i,r=e)}f[e]=r}return l.forEach(e=>n.disposeIntermediateTensorInfo(e)),n.makeTensorInfo(u,`int32`,f)}let Pu={kernelName:xe,backendName:`cpu`,kernelFunc:Nu};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Fu(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{axis:a}=r;q(i,`argMin`);let o=S(a,i.shape),s=uo(o,i.shape.length),c=i,l=[];s!=null&&(c=ml({inputs:{x:i},backend:n,attrs:{perm:s}}),l.push(c),o=po(o.length,c.shape.length)),o=[o[0]],lo(`argMin`,o,c.shape.length);let[u,d]=so(c.shape,o),f=I(_(u),`int32`),p=_(d),m=n.data.get(c.dataId).values;for(let e=0;e<f.length;++e){let t=e*p,n=m[t],r=0;for(let e=0;e<p;++e){let i=m[t+e];i<n&&(n=i,r=e)}f[e]=r}return l.forEach(e=>n.disposeIntermediateTensorInfo(e)),n.makeTensorInfo(u,`int32`,f)}let Iu={kernelName:Se,backendName:`cpu`,kernelFunc:Fu},Lu={kernelName:Ce,backendName:`cpu`,kernelFunc:J(Ce,e=>Math.asin(e))},Ru={kernelName:we,backendName:`cpu`,kernelFunc:J(we,e=>Math.asinh(e))},zu={kernelName:Te,backendName:`cpu`,kernelFunc:J(Te,e=>Math.atan(e))},Bu={kernelName:De,backendName:`cpu`,kernelFunc:Ac(De,yc((e,t)=>Math.atan2(e,t)))},Vu={kernelName:Ee,backendName:`cpu`,kernelFunc:J(Ee,e=>Math.atanh(e))}
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the License);
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an AS IS BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the License);
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an AS IS BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the License);
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an AS IS BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the License);
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an AS IS BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the License);
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an AS IS BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Hu(e,t,n,r,i,a){let o=i.strideHeight,s=i.strideWidth,c=i.dilationHeight,l=i.dilationWidth,u=i.effectiveFilterHeight,d=i.effectiveFilterWidth,f=i.padInfo.top,p=i.padInfo.left,m=a===`max`?-1/0:1/0,h=W(i.outShape,n),g=h.values,_=i.outShape[1]*i.outShape[2]*i.outShape[3],v=i.outShape[2]*i.outShape[3],y=i.outShape[3];for(let t=0;t<i.batchSize;++t){let n=t*_,h=t*r[0];for(let t=0;t<i.inChannels;++t)for(let _=0;_<i.outHeight;++_){let b=_*o-f,x=Math.max(0,b),S=Math.min(i.inHeight,u+b),C=n+_*v;for(let n=0;n<i.outWidth;++n){let o=n*s-p,u=Math.max(0,o),f=Math.min(i.inWidth,d+o),_=m,v=0,b=0;for(let n=x;n<S;n+=c){let i=h+n*r[1];for(let n=u;n<f;n+=l){let o=e[i+n*r[2]+t];a===`max`&&o>_?_=o:a===`avg`&&(v+=o,b++)}if(isNaN(_))break}let w=C+n*y+t;g[w]=a===`avg`?v/b:_}}}return h}function Uu(e,t,n,r,i=!1,a=!1){let o=W(r.outShape,`int32`),s=r.strideHeight,c=r.strideWidth,l=r.dilationHeight,u=r.dilationWidth,d=r.effectiveFilterHeight,f=r.effectiveFilterWidth,p=r.padInfo.top,m=r.padInfo.left,h=W(t,n,e);for(let e=0;e<r.batchSize;++e)for(let t=0;t<r.inChannels;++t)for(let n=0;n<r.outHeight;++n){let g=n*s-p,_=g;for(;_<0;)_+=l;let v=Math.min(r.inHeight,d+g);for(let s=0;s<r.outWidth;++s){let d=s*c-m,p=d;for(;p<0;)p+=u;let y=Math.min(r.inWidth,f+d),b=-1/0,x=-1;for(let n=_;n<v;n+=l){let o=n-g;for(let s=p;s<y;s+=u){let c=s-d,l=h.get(e,n,s,t);l>b&&(b=l,x=i?a?((e*r.inHeight+n)*r.inWidth+s)*r.inChannels+t:(n*r.inWidth+s)*r.inChannels+t:o*f+c)}}o.set(x,e,n,s,t)}}return o}function Wu(e,t,n,r,i,a){let o=i.strideDepth,s=i.strideHeight,c=i.strideWidth,l=i.dilationDepth,u=i.dilationHeight,d=i.dilationWidth,f=i.effectiveFilterDepth,p=i.effectiveFilterHeight,m=i.effectiveFilterWidth,h=i.padInfo.front,g=i.padInfo.top,_=i.padInfo.left,v=a===`max`?-1/0:1/0,y=W(i.outShape,n),b=y.values,x=i.outShape[1]*i.outShape[2]*i.outShape[3]*i.outShape[4],S=i.outShape[2]*i.outShape[3]*i.outShape[4],C=i.outShape[3]*i.outShape[4],w=i.outShape[4];for(let t=0;t<i.batchSize;++t){let n=t*x,y=t*r[0];for(let t=0;t<i.inChannels;++t)for(let x=0;x<i.outDepth;++x){let T=x*o-h,E=T;for(;E<0;)E+=l;let D=Math.min(i.inDepth,f+T),O=n+x*S;for(let n=0;n<i.outHeight;++n){let o=n*s-g,f=o;for(;f<0;)f+=u;let h=Math.min(i.inHeight,p+o),x=O+n*C;for(let n=0;n<i.outWidth;++n){let o=n*c-_,s=o;for(;s<0;)s+=d;let p=Math.min(i.inWidth,m+o),g=x+n*w,S=v,C=0,T=0;for(let n=E;n<D;n+=l){let i=y+n*r[1];for(let n=f;n<h;n+=u){let o=i+n*r[2];for(let n=s;n<p;n+=d){let i=e[o+n*r[3]+t];if(a===`max`&&i>S?S=i:a===`avg`&&(C+=i,T++),isNaN(S))break}if(isNaN(S))break}if(isNaN(S))break}let O=g+t;b[O]=a===`avg`?C/Math.max(T,1):S}}}}return y}function Gu(e,t){let n=W(t.outShape,`int32`),r=t.strideDepth,i=t.strideHeight,a=t.strideWidth,o=t.dilationDepth,s=t.dilationHeight,c=t.dilationWidth,l=t.effectiveFilterDepth,u=t.effectiveFilterHeight,d=t.effectiveFilterWidth,f=t.padInfo.front,p=t.padInfo.top,m=t.padInfo.left;for(let h=0;h<t.batchSize;++h)for(let g=0;g<t.inChannels;++g)for(let _=0;_<t.outDepth;++_){let v=_*r-f,y=v;for(;y<0;)y+=o;let b=Math.min(t.inDepth,l+v);for(let r=0;r<t.outHeight;++r){let l=r*i-p,f=l;for(;f<0;)f+=s;let x=Math.min(t.inHeight,u+l);for(let i=0;i<t.outWidth;++i){let p=i*a-m,S=p;for(;S<0;)S+=c;let C=Math.min(t.inWidth,d+p),w=-1/0,T=-1;for(let t=y;t<b;t+=o){let n=t-v;for(let r=f;r<x;r+=s){let i=r-l;for(let a=S;a<C;a+=c){let o=a-p,s=e.get(h,t,r,a,g);s>=w&&(w=s,T=n*u*d+i*u+o)}}}n.set(T,h,_,r,i,g)}}}return n}
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Ku(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t;q(i,`avgPool`);let{filterSize:a,strides:o,pad:s,dimRoundingMode:c}=r;m(Ra(o,1),()=>`Error in avgPool: Either strides or dilations must be 1. Got strides ${o} and dilations '1'`);let l=wa(i.shape,a,o,1,s,c),u;if(l.filterWidth===1&&l.filterHeight===1&&v(l.inShape,l.outShape))u=Cc({inputs:{x:i},backend:n});else{let e=n.data.get(i.dataId).values,t=P(i.shape),r=Hu(e,i.shape,i.dtype,t,l,`avg`);u=n.makeTensorInfo(l.outShape,i.dtype,r.values)}return u}let qu={kernelName:Oe,backendName:`cpu`,kernelFunc:Ku};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Ju(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{filterSize:a,strides:o,pad:s,dimRoundingMode:c,dataFormat:l}=r;q(i,`avgPool3d`);let u=Ta(i.shape,a,o,1,s,c,l),d=n.data.get(i.dataId).values,f=Wu(d,i.shape,i.dtype,P(i.shape),u,`avg`);return n.makeTensorInfo(f.shape,`float32`,f.values)}let Yu={kernelName:Ae,backendName:`cpu`,kernelFunc:Ju};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Xu(e){let{inputs:t,backend:n,attrs:r}=e,{dy:i,input:a}=t,{filterSize:o,strides:s,pad:c,dimRoundingMode:l}=r;q([i,a],`avgPool3DGrad`);let u=Ta(a.shape,o,s,1,c,l),d=u.strideDepth,f=u.strideHeight,p=u.strideWidth,m=u.filterDepth,h=u.filterHeight,g=u.filterWidth,_=u.dilationDepth,v=u.dilationHeight,y=u.dilationWidth,b=u.effectiveFilterDepth,x=u.effectiveFilterHeight,S=u.effectiveFilterWidth,C=b-1-u.padInfo.front,w=S-1-u.padInfo.left,T=x-1-u.padInfo.top,E=W(a.shape,`float32`),D=1/(m*h*g),O=n.bufferSync(i);for(let e=0;e<u.batchSize;++e)for(let t=0;t<u.inChannels;++t)for(let n=0;n<u.inDepth;++n)for(let r=0;r<u.inHeight;++r)for(let i=0;i<u.inWidth;++i){let a=n-C,o=r-T,s=i-w,c=0;for(let n=0;n<b;n+=_){let r=(a+n)/d;if(!(r<0||r>=u.outDepth||Math.floor(r)!==r))for(let n=0;n<x;n+=v){let i=(o+n)/f;if(!(i<0||i>=u.outHeight||Math.floor(i)!==i))for(let n=0;n<S;n+=y){let a=(s+n)/p;if(a<0||a>=u.outWidth||Math.floor(a)!==a)continue;let o=O.get(e,r,i,a,t);c+=o}}}E.set(c*D,e,n,r,i,t)}return n.makeTensorInfo(E.shape,E.dtype,E.values)}let Zu={kernelName:je,backendName:`cpu`,kernelFunc:Xu};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Qu(e){let{inputs:t,backend:n,attrs:r}=e,{dy:i,input:a}=t,o=a;q([i,a],`avgPoolGrad`);let{filterSize:s,strides:c,pad:l}=r,u=wa(o.shape,s,c,1,l),d=u.strideHeight,f=u.strideWidth,p=u.filterHeight,m=u.filterWidth,h=u.dilationHeight,g=u.dilationWidth,_=u.effectiveFilterHeight,v=u.effectiveFilterWidth,y=v-1-u.padInfo.left,b=_-1-u.padInfo.top,x=W(o.shape,`float32`),S=1/(p*m),C=n.data.get(i.dataId).values,w=W(i.shape,`float32`,C);for(let e=0;e<u.batchSize;++e)for(let t=0;t<u.inChannels;++t)for(let n=0;n<u.inHeight;++n)for(let r=0;r<u.inWidth;++r){let i=n-b,a=r-y,o=0;for(let n=0;n<_;n+=h){let r=(i+n)/d;if(!(r<0||r>=u.outHeight||Math.floor(r)!==r))for(let n=0;n<v;n+=g){let i=(a+n)/f;if(i<0||i>=u.outWidth||Math.floor(i)!==i)continue;let s=w.get(e,r,i,t);o+=s}}x.set(o*S,e,n,r,t)}return n.makeTensorInfo(x.shape,x.dtype,x.values)}let $u={kernelName:ke,backendName:`cpu`,kernelFunc:Qu};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function ed(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,scale:a,offset:o,mean:s,variance:c}=t;m(s.shape.length===c.shape.length,()=>`Batch normalization gradient requires mean and variance to have equal ranks.`),m(o==null||s.shape.length===o.shape.length,()=>`Batch normalization gradient requires mean and offset to have equal ranks.`),m(a==null||s.shape.length===a.shape.length,()=>`Batch normalization gradient requires mean and scale to have equal ranks.`),q([i,s,c,a,o],`batchNorm`);let{varianceEpsilon:l}=r;l??=.001;let u=n.data.get(i.dataId).values,d=n.data.get(s.dataId).values,f=n.data.get(c.dataId).values,p=a?n.data.get(a.dataId).values:new Float32Array([1]),h=o?n.data.get(o.dataId).values:new Float32Array([0]),g=new Float32Array(u.length),_=h.length,v=p.length,y=f.length,b=d.length,x=0,S=0,C=0,w=0;for(let e=0;e<u.length;++e)g[e]=h[x++]+(u[e]-d[S++])*p[C++]/Math.sqrt(f[w++]+l),x>=_&&(x=0),S>=b&&(S=0),C>=v&&(C=0),w>=y&&(w=0);return n.makeTensorInfo(i.shape,i.dtype,g)}let td={kernelName:ft,backendName:`cpu`,kernelFunc:ed};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function nd(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{blockShape:a,crops:o}=r;q([i],`batchToSpaceND`);let s=a.reduce((e,t)=>e*t),c=Ns(i.shape,a,s),l=Ps(c.length,a.length),u=Fs(i.shape,a,s),d=Is(o,a.length),f=Ls(u,o,a.length),p=yu({inputs:{x:i},backend:n,attrs:{shape:c}}),m=ml({inputs:{x:p},backend:n,attrs:{perm:l}}),h=yu({inputs:{x:m},backend:n,attrs:{shape:u}}),g=Bl({inputs:{x:h},backend:n,attrs:{begin:d,size:f}});return n.disposeIntermediateTensorInfo(p),n.disposeIntermediateTensorInfo(m),n.disposeIntermediateTensorInfo(h),g}let rd={kernelName:Ne,backendName:`cpu`,kernelFunc:nd};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function id(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,weights:a}=t,{size:o}=r,s=n.data.get(i.dataId).values,c=n.data.get(a.dataId).values,l=Pc(s,c,a.dtype,a.shape,o);return n.makeTensorInfo([o],a.dtype,l)}let ad={kernelName:Pe,backendName:`cpu`,kernelFunc:id};
/**
* @license
* Copyright 2021 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function od(e){let{inputs:t,backend:n}=e,{s0:r,s1:i}=t,a=n.data.get(r.dataId).values,o=n.data.get(i.dataId).values,s=no(Array.from(a),Array.from(o));return n.makeTensorInfo([s.length],`int32`,Int32Array.from(s))}let sd={kernelName:Ie,backendName:`cpu`,kernelFunc:od},cd={kernelName:ze,backendName:`cpu`,kernelFunc:J(ze,(e,t)=>{
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the License);
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an AS IS BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let n=t;return e>n.clipValueMax?n.clipValueMax:e<n.clipValueMin?n.clipValueMin:e})},ld={kernelName:Be,backendName:`cpu`,kernelFunc:e=>{
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the License);
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an AS IS BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let{x:t}=e.inputs,n=e.backend,r=new Float32Array(_(t.shape)),i=n.data.get(t.dataId),a=i.complexTensorInfos.real,o=i.complexTensorInfos.imag,s=n.data.get(a.dataId).values,c=n.data.get(o.dataId).values;for(let e=0;e<s.length;e++){let t=s[e],n=c[e];r[e]=Math.hypot(t,n)}return n.makeOutput(r,t.shape,`float32`)}};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function ud(e){let{inputs:t,backend:n}=e,{input:r}=t,i=n.data.get(r.dataId).complexTensorInfos.imag,a=n.data.get(i.dataId).values;return n.makeTensorInfo(i.shape,i.dtype,a)}let dd={kernelName:`Imag`,backendName:`cpu`,kernelFunc:ud};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function fd(e){let{inputs:t,backend:n,attrs:r}=e,{axis:i}=r,a=S(i,t[0].shape)[0];Ts(t.map(e=>e.shape),a);let o=Es(t.map(e=>e.shape),a);if(_(o)===0)return n.makeTensorInfo(o,t[0].dtype,[]);let s=t.filter(e=>_(e.shape)>0);if(s.length===1)return Cc({inputs:{x:s[0]},backend:n});if(s[0].dtype===`complex64`){let e=s.map(e=>Tc({inputs:{input:e},backend:n})),t=s.map(e=>ud({inputs:{input:e},backend:n})),r=fd({inputs:e,backend:n,attrs:{axis:a}}),i=fd({inputs:t,backend:n,attrs:{axis:a}}),o=bc({inputs:{real:r,imag:i},backend:n});return e.forEach(e=>n.disposeIntermediateTensorInfo(e)),t.forEach(e=>n.disposeIntermediateTensorInfo(e)),n.disposeIntermediateTensorInfo(r),n.disposeIntermediateTensorInfo(i),o}let c=s.map(e=>{let t=[-1,_(e.shape.slice(a))];return yu({inputs:{x:e},backend:n,attrs:{shape:t}})}),l=c.map(e=>({vals:n.data.get(e.dataId).values,shape:e.shape}));o=Es(c.map(e=>e.shape),1);let u=c[0].shape[0]===1,d=Bc(l,o,t[0].dtype,u),f=Es(s.map(e=>e.shape),a),p=n.makeTensorInfo(f,t[0].dtype,d);return c.forEach(e=>n.disposeIntermediateTensorInfo(e)),p}let pd={kernelName:Ve,backendName:`cpu`,kernelFunc:fd};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function md(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,filter:a}=t,{strides:o,pad:s,dataFormat:c,dilations:l,dimRoundingMode:u}=r;q([i,a],`conv2d`);let d=Ba(c),f=Ea(i.shape,a.shape,o,l,s,u,!1,d),p=f.filterHeight,m=f.filterWidth,h=f.dilationHeight,g=f.dilationWidth,_=f.padInfo.left,v=f.padInfo.top,y=f.dataFormat===`channelsLast`,b=new Vr(f.outShape,i.dtype),x=P(i.shape),S=P(a.shape),C=x[0],w=y?x[1]:x[2],T=y?x[2]:1,E=y?1:x[1],D=b.strides[0],O=y?b.strides[1]:b.strides[2],k=y?b.strides[2]:1,A=y?1:b.strides[1],j=n.data.get(i.dataId).values,M=n.data.get(a.dataId).values,N=b.values;for(let e=0;e<f.batchSize;++e){let t=e*C,n=e*D;for(let e=0;e<f.outHeight;++e){let r=n+e*O,i=e*f.strideHeight-v;for(let e=0;e<p;++e){let n=i+e*h;if(n<0||n>=f.inHeight)continue;let a=e*S[0],o=t+n*w;for(let e=0;e<f.outWidth;++e){let t=r+e*k,n=e*f.strideWidth-_;for(let e=0;e<m;++e){let r=n+e*g;if(r<0||r>=f.inWidth)continue;let i=a+e*S[1],s=o+r*T,c=i;for(let e=0;e<f.inChannels;++e){let n=j[s+e*E];for(let e=0;e<f.outChannels;++e)N[t+e*A]+=n*M[c+e];c+=f.outChannels}}}}}}return n.makeTensorInfo(b.shape,b.dtype,N)}let hd={kernelName:He,backendName:`cpu`,kernelFunc:md};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function gd(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,dy:a}=t,{strides:o,pad:s,dataFormat:c,dimRoundingMode:l,filterShape:u}=r;q([i,a],`conv2dBackpropFilter`);let d=Ba(c),f=Ea(i.shape,u,o,1,s,l,!1,d),{strideHeight:p,strideWidth:m,filterHeight:h,filterWidth:g}=f,_=f.dataFormat===`channelsLast`,v=new Vr(f.filterShape,`float32`),y=f.padInfo.left,b=f.padInfo.top,x=n.data.get(i.dataId).values,S=n.data.get(a.dataId).values,C=new Vr(i.shape,i.dtype,x),w=new Vr(a.shape,a.dtype,S);for(let e=0;e<h;++e){let t=Math.max(0,Math.ceil((b-e)/p)),n=Math.min(f.outHeight,(f.inHeight+b-e)/p);for(let r=0;r<g;++r){let i=Math.max(0,Math.ceil((y-r)/m)),a=Math.min(f.outWidth,(f.inWidth+y-r)/m);for(let o=0;o<f.inChannels;++o)for(let s=0;s<f.outChannels;++s){let c=0;for(let l=0;l<f.batchSize;++l)for(let u=t;u<n;++u){let t=e+u*p-b;for(let e=i;e<a;++e){let n=r+e*m-y;c+=_?C.get(l,t,n,o)*w.get(l,u,e,s):C.get(l,o,t,n)*w.get(l,s,u,e)}}v.set(c,e,r,o,s)}}}return n.makeTensorInfo(v.shape,v.dtype,v.values)}let _d={kernelName:`Conv2DBackpropFilter`,backendName:`cpu`,kernelFunc:gd};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function vd(e){let{inputs:t,backend:n,attrs:r}=e,{dy:i,filter:a}=t,{inputShape:o,strides:s,pad:c,dataFormat:l,dimRoundingMode:u}=r;q([i,a],`conv2dBackpropInput`);let d=P(a.shape),f=P(i.shape),p=Ba(l),m=Ea(o,a.shape,s,1,c,u,!1,p),h=new Vr(m.inShape,`float32`),g=h.values,_=n.data.get(i.dataId).values,v=n.data.get(a.dataId).values,[y,b,x]=d,{batchSize:S,filterHeight:C,filterWidth:w,inChannels:T,inHeight:E,inWidth:D,outChannels:O,outHeight:k,outWidth:A,strideHeight:j,strideWidth:M}=m;p=m.dataFormat;let N=C-1-m.padInfo.top,ee=w-1-m.padInfo.left,te=p===`channelsLast`,F=h.strides[0],ne=te?h.strides[1]:h.strides[2],re=te?h.strides[2]:1,I=te?1:h.strides[1],ie=f[0],L=te?f[1]:f[2],R=te?f[2]:1,ae=te?1:f[1];for(let e=0;e<S;++e)for(let t=0;t<T;++t)for(let n=0;n<E;++n){let r=n-N,i=Math.max(0,Math.ceil(r/j)),a=Math.min(k,(C+r)/j);for(let o=0;o<D;++o){let s=o-ee,c=Math.max(0,Math.ceil(s/M)),l=Math.min(A,(w+s)/M),u=0;for(let n=i;n<a;++n){let i=n*j-r;for(let r=c;r<l;++r){let a=r*M-s,o=ie*e+L*n+R*r,c=y*(C-1-i)+b*(w-1-a)+x*t;for(let e=0;e<O;++e){let t=_[o+ae*e],n=v[c+e];u+=t*n}}}let d=F*e+ne*n+re*o+I*t;g[d]=u}}return n.makeTensorInfo(h.shape,h.dtype,h.values)}let yd={kernelName:Ue,backendName:`cpu`,kernelFunc:vd};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function bd(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,filter:a}=t,{strides:o,pad:s,dilations:c}=r;q([i,a],`conv3d`);let l=Da(i.shape,a.shape,o,c,s),{filterDepth:u,filterHeight:d,filterWidth:f,dilationDepth:p,dilationHeight:m,dilationWidth:h,padInfo:g}=l,_=g.front,v=g.left,y=g.top,b=new Vr(l.outShape,i.dtype),x=n.data.get(i.dataId).values,S=n.data.get(a.dataId).values,C=b.values,w=P(i.shape),T=P(a.shape);for(let e=0;e<l.batchSize;++e){let t=e*w[0],n=e*b.strides[0];for(let e=0;e<l.outDepth;++e){let r=n+e*b.strides[1],i=e*l.strideDepth-_;for(let e=0;e<u;++e){let n=i+e*p;if(n<0||n>=l.inDepth)continue;let a=e*T[0],o=t+n*w[1];for(let e=0;e<l.outHeight;++e){let t=r+e*b.strides[2],n=e*l.strideHeight-y;for(let e=0;e<d;++e){let r=n+e*m;if(r<0||r>=l.inHeight)continue;let i=a+e*T[1],s=o+r*w[2];for(let e=0;e<l.outWidth;++e){let n=t+e*l.outChannels,r=e*l.strideWidth-v;for(let e=0;e<f;++e){let t=r+e*h;if(t<0||t>=l.inWidth)continue;let a=i+e*T[2],o=s+t*l.inChannels,c=a;for(let e=0;e<l.inChannels;++e){let t=x[o+e];for(let e=0;e<l.outChannels;++e)C[n+e]+=t*S[c+e];c+=l.outChannels}}}}}}}}return n.makeTensorInfo(b.shape,b.dtype,b.values)}let xd={kernelName:We,backendName:`cpu`,kernelFunc:bd};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Sd(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,dy:a}=t,{strides:o,pad:s,filterShape:c}=r;q([i,a],`conv3dBackpropFilterV2`);let l=P(i.shape),u=P(a.shape),d=Da(i.shape,c,o,1,s),f=d.strideDepth,p=d.strideHeight,m=d.strideWidth,h=d.filterDepth,g=d.filterHeight,_=d.filterWidth,v=new Vr(d.filterShape,`float32`),y=v.values,[b,x,S,C]=v.strides,w=n.data.get(a.dataId).values,[T,E,D,O]=u,k=n.data.get(i.dataId).values,[A,j,M,N]=l,ee=d.padInfo.front,te=d.padInfo.left,F=d.padInfo.top;for(let e=0;e<h;++e){let t=Math.max(0,Math.ceil((ee-e)/f)),n=Math.min(d.outDepth,(d.inDepth+ee-e)/f),r=e*b;for(let i=0;i<g;++i){let a=Math.max(0,Math.ceil((F-i)/p)),o=Math.min(d.outHeight,(d.inHeight+F-i)/p),s=i*x+r;for(let r=0;r<_;++r){let c=Math.max(0,Math.ceil((te-r)/m)),l=Math.min(d.outWidth,(d.inWidth+te-r)/m),u=r*S+s;for(let s=0;s<d.inChannels;++s){let h=s*C+u;for(let u=0;u<d.outChannels;++u){let g=0;for(let h=0;h<d.batchSize;++h){let d=h*A,_=h*T;for(let h=t;h<n;++h){let t=(e+h*f-ee)*j+d,n=h*E+_;for(let e=a;e<o;++e){let a=(i+e*p-F)*M+t,o=e*D+n;for(let e=c;e<l;++e){let t=(r+e*m-te)*N+a,n=e*O+o;g+=k[t+s]*w[n+u]}}}}y[h+u]=g}}}}}return n.makeTensorInfo(v.shape,v.dtype,v.values)}let Cd={kernelName:B,backendName:`cpu`,kernelFunc:Sd};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function wd(e){let{inputs:t,backend:n,attrs:r}=e,{dy:i,filter:a}=t,{pad:o,strides:s,inputShape:c}=r;q([i],`conv3dBackpropInputV2`);let l=P(i.shape),u=P(a.shape),d=Da(c,a.shape,s,1,o),f=new Vr(d.inShape,`float32`),p=f.values,[m,h,g,_]=f.strides,v=n.data.get(i.dataId).values,[y,b,x,S]=l,C=n.data.get(a.dataId).values,[w,T,E,D]=u,{batchSize:O,filterDepth:k,filterHeight:A,filterWidth:j,inChannels:M,inDepth:N,inHeight:ee,inWidth:te,outChannels:F,outDepth:ne,outHeight:re,outWidth:I,strideDepth:ie,strideHeight:L,strideWidth:R}=d,ae=k-1-d.padInfo.front,oe=A-1-d.padInfo.top,se=j-1-d.padInfo.left;for(let e=0;e<O;++e)for(let t=0;t<M;++t)for(let n=0;n<N;++n){let r=n-ae,i=Math.max(0,Math.ceil(r/ie)),a=Math.min(ne,(k+r)/ie);for(let o=0;o<ee;++o){let s=o-oe,c=Math.max(0,Math.ceil(s/L)),l=Math.min(re,(A+s)/L);for(let u=0;u<te;++u){let d=u-se,f=Math.max(0,Math.ceil(d/R)),O=Math.min(I,(j+d)/R),M=0;for(let n=i;n<a;++n){let i=n*ie-r;for(let r=c;r<l;++r){let a=r*L-s;for(let o=f;o<O;++o){let s=o*R-d,c=y*e+b*n+x*r+S*o,l=w*(k-1-i)+T*(A-1-a)+E*(j-1-s)+D*t;for(let e=0;e<F;++e){let t=v[c+e],n=C[l+e];M+=t*n}}}}p[m*e+h*n+g*o+_*u+t]=M}}}return n.makeTensorInfo(f.shape,f.dtype,f.values)}let Td={kernelName:Ge,backendName:`cpu`,kernelFunc:wd},Ed={kernelName:`Cos`,backendName:`cpu`,kernelFunc:J(`Cos`,e=>Math.cos(e))},Dd={kernelName:Ke,backendName:`cpu`,kernelFunc:J(Ke,e=>Math.cosh(e))}
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the License);
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an AS IS BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Od(e){let{inputs:t,backend:n,attrs:r}=e,{image:i,boxes:a,boxInd:o}=t,{cropSize:s,method:c,extrapolationValue:l}=r,[u,d,f,p]=i.shape,m=a.shape[0],[h,g]=s,_=W([m,h,g,p],`float32`),v=n.data.get(a.dataId).values,y=n.data.get(o.dataId).values,b=n.data.get(i.dataId).values,x=P(i.shape),S=P(_.shape);for(let e=0;e<m;e++){let t=e*4,n=v[t],r=v[t+1],i=v[t+2],a=v[t+3],o=y[e];if(o>=u)continue;let s=h>1?(i-n)*(d-1)/(h-1):0,m=g>1?(a-r)*(f-1)/(g-1):0;for(let t=0;t<h;t++){let u=h>1?n*(d-1)+t*s:.5*(n+i)*(d-1);if(u<0||u>d-1){for(let n=0;n<g;n++)for(let r=0;r<p;r++){let i=r+n*S[2]+t*S[1]+e*S[0];_.values[i]=l}continue}if(c===`bilinear`){let n=Math.floor(u),i=Math.ceil(u),s=u-n;for(let c=0;c<g;c++){let u=g>1?r*(f-1)+c*m:.5*(r+a)*(f-1);if(u<0||u>f-1){for(let n=0;n<p;n++){let r=n+c*S[2]+t*S[1]+e*S[0];_.values[r]=l}continue}let d=Math.floor(u),h=Math.ceil(u),v=u-d;for(let r=0;r<p;r++){let a=r+d*x[2]+n*x[1]+o*x[0],l=b[a];a=r+h*x[2]+n*x[1]+o*x[0];let u=b[a];a=r+d*x[2]+i*x[1]+o*x[0];let f=b[a];a=r+h*x[2]+i*x[1]+o*x[0];let p=b[a],m=l+(u-l)*v,g=f+(p-f)*v;a=r+c*S[2]+t*S[1]+e*S[0],_.values[a]=m+(g-m)*s}}}else for(let n=0;n<g;++n){let i=g>1?r*(f-1)+n*m:.5*(r+a)*(f-1);if(i<0||i>f-1){for(let r=0;r<p;r++){let i=r+n*S[2]+t*S[1]+e*S[0];_.values[i]=l}continue}let s=Math.round(i),c=Math.round(u);for(let r=0;r<p;r++){let i=r+s*x[2]+c*x[1]+o*x[0],a=r+n*S[2]+t*S[1]+e*S[0];_.values[a]=b[i]}}}}return n.makeTensorInfo(_.shape,_.dtype,_.values)}let kd={kernelName:Ye,backendName:`cpu`,kernelFunc:Od};
/**
* @license
* Copyright 2022 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Ad(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{axis:a,exclusive:o,reverse:s}=r;q(i,`cumprod`);let c=uo([a],i.shape.length),l=i;c!=null&&(l=ml({inputs:{x:i},backend:n,attrs:{perm:c}}));let u=po(1,i.shape.length)[0];if(u!==l.shape.length-1)throw Error(`backend.cumprod in CPU expects an inner-most axis=${l.shape.length-1} but got axis=${u}`);let d=ti(l.dtype,`int32`),f=re(_(l.shape),d),p=n.data.get(l.dataId).values,m=l.shape[l.shape.length-1],h=s?(e,t)=>e+m-t-1:(e,t)=>e+t;for(let e=0;e<p.length;e+=m)for(let t=0;t<m;t++){let n=h(e,t);if(t===0)f[n]=o?1:p[n];else{let r=h(e,t-1);f[n]=o?p[r]*f[r]:p[n]*f[r]}}let g=n.makeTensorInfo(l.shape,d,f);if(c!=null){let e=fo(c),t=ml({inputs:{x:g},backend:n,attrs:{perm:e}});return n.disposeIntermediateTensorInfo(g),n.disposeIntermediateTensorInfo(l),t}return g}let jd={kernelName:qe,backendName:`cpu`,kernelFunc:Ad};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Md(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{axis:a,exclusive:o,reverse:s}=r;q(i,`cumsum`);let c=uo([a],i.shape.length),l=i;c!=null&&(l=ml({inputs:{x:i},backend:n,attrs:{perm:c}}));let u=po(1,i.shape.length)[0];if(u!==l.shape.length-1)throw Error(`backend.cumsum in CPU expects an inner-most axis=${l.shape.length-1} but got axis=${u}`);let d=ti(l.dtype,`int32`),f=I(_(l.shape),d),p=n.data.get(l.dataId).values,m=l.shape[l.shape.length-1],h=s?(e,t)=>e+m-t-1:(e,t)=>e+t;for(let e=0;e<p.length;e+=m)for(let t=0;t<m;t++){let n=h(e,t);if(t===0)f[n]=o?0:p[n];else{let r=h(e,t-1);f[n]=o?p[r]+f[r]:p[n]+f[r]}}let g=n.makeTensorInfo(l.shape,d,f);if(c!=null){let e=fo(c),t=ml({inputs:{x:g},backend:n,attrs:{perm:e}});return n.disposeIntermediateTensorInfo(g),n.disposeIntermediateTensorInfo(l),t}return g}let Nd={kernelName:Je,backendName:`cpu`,kernelFunc:Md};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Pd(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,weights:a}=t,{size:o,binaryOutput:s}=r;if(i.shape.length===1){let e=n.data.get(i.dataId).values,t=n.data.get(a.dataId).values,r=Pc(e,t,a.dtype,a.shape,o);return n.makeTensorInfo([o],a.dtype,r)}if(i.shape.length===2){let e=Fc(n.bufferSync(i),n.bufferSync(a),o,s);return n.makeTensorInfo(e.shape,a.dtype,e.values)}throw Error(`Error in denseBincount: input must be at most rank 2, but got rank${i.shape.length}.`)}let Fd={kernelName:Xe,backendName:`cpu`,kernelFunc:Pd};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Id(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{blockSize:a,dataFormat:o}=r;m(o===`NHWC`,()=>`Only NHWC dataFormat supported on CPU for depthToSpace. Got ${o}`);let s=i.shape[0],c=i.shape[1],l=i.shape[2],u=i.shape[3],d=c*a,f=l*a,p=u/(a*a),h=n.data.get(i.dataId).values,g=new Float32Array(s*d*f*p),_=0;for(let e=0;e<s;++e)for(let t=0;t<d;++t){let n=Math.floor(t/a),r=t%a;for(let t=0;t<f;++t){let i=Math.floor(t/a),o=t%a,s=(r*a+o)*p;for(let t=0;t<p;++t){let r=t+s+u*(i+l*(n+c*e));g[_++]=h[r]}}}return n.makeTensorInfo([s,d,f,p],i.dtype,g)}let Ld={kernelName:Ze,backendName:`cpu`,kernelFunc:Id};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Rd(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,filter:a}=t,{strides:o,pad:s,dilations:c,dimRoundingMode:l}=r;q([i,a],`depthwiseConv2DNative`);let u=P(i.shape),d=P(a.shape),f=c;f??=[1,1],m(Ra(o,f),()=>`Error in depthwiseConv2d: Either strides or dilations must be 1. Got strides ${o} and dilations '${f}'`);let p=Ea(i.shape,a.shape,o,f,s,l,!0),{filterHeight:h,filterWidth:g,dilationHeight:_,dilationWidth:v,padInfo:y}=p,b=y.left,x=y.top,S=p.outChannels/p.inChannels,C=new Vr(p.outShape,i.dtype),w=n.data.get(i.dataId).values,T=n.data.get(a.dataId).values,E=C.values;for(let e=0;e<p.batchSize;++e){let t=e*u[0],n=e*C.strides[0];for(let e=0;e<p.outHeight;++e){let r=n+e*C.strides[1],i=e*p.strideHeight-x;for(let e=0;e<h;++e){let n=i+e*_;if(n<0||n>=p.inHeight)continue;let a=e*d[0],o=t+n*u[1];for(let e=0;e<p.outWidth;++e){let t=r+e*C.strides[2],n=e*p.strideWidth-b;for(let e=0;e<g;++e){let r=n+e*v;if(r<0||r>=p.inWidth)continue;let i=a+e*d[1],s=o+r*p.inChannels,c=t,l=i;for(let e=0;e<p.inChannels;++e){let t=w[s+e];for(let e=0;e<S;++e)E[c+e]+=t*T[l+e];c+=S,l+=S}}}}}}return n.makeTensorInfo(C.shape,C.dtype,C.values)}let zd={kernelName:Qe,backendName:`cpu`,kernelFunc:Rd};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Bd(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,dy:a}=t,{strides:o,dilations:s,pad:c,dimRoundingMode:l,filterShape:u}=r;q([i,a],`depthwiseConv2dNativeBackpropFilter`);let d=Ea(i.shape,u,o,s,c,l,!0),{strideHeight:f,strideWidth:p,filterHeight:m,filterWidth:h}=d,g=new Vr(d.filterShape,`float32`),_=d.padInfo.left,v=d.padInfo.top,y=d.outChannels/d.inChannels,b=n.data.get(i.dataId).values,x=new Vr(i.shape,i.dtype,b),S=n.data.get(a.dataId).values,C=new Vr(a.shape,a.dtype,S);for(let e=0;e<m;++e){let t=Math.max(0,Math.ceil((v-e)/f)),n=Math.min(d.outHeight,(d.inHeight+v-e)/f);for(let r=0;r<h;++r){let i=Math.max(0,Math.ceil((_-r)/p)),a=Math.min(d.outWidth,(d.inWidth+_-r)/p);for(let o=0;o<d.outChannels;++o){let s=Math.trunc(o/y),c=o%y,l=0;for(let c=0;c<d.batchSize;++c)for(let u=t;u<n;++u){let t=e+u*f-v;for(let e=i;e<a;++e){let n=r+e*p-_;l+=x.get(c,t,n,s)*C.get(c,u,e,o)}}g.set(l,e,r,s,c)}}}return n.makeTensorInfo(g.shape,g.dtype,g.values)}let Vd={kernelName:`DepthwiseConv2dNativeBackpropFilter`,backendName:`cpu`,kernelFunc:Bd};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Hd(e){let{inputs:t,backend:n,attrs:r}=e,{dy:i,filter:a}=t,{strides:o,dilations:s,pad:c,dimRoundingMode:l,inputShape:u}=r;q([i,a],`depthwiseConv2DNativeBackpropInput`);let d=P(i.shape),f=P(a.shape),p=Ea(u,a.shape,o,s,c,l,!0),m=new Vr(p.inShape,`float32`),h=m.values,[g,_,v]=m.strides,y=n.data.get(i.dataId).values,[b,x,S]=d,C=n.data.get(a.dataId).values,[w,T,E]=f,{batchSize:D,filterHeight:O,filterWidth:k,inChannels:A,inHeight:j,inWidth:M,outChannels:N,outHeight:ee,outWidth:te,strideHeight:F,strideWidth:ne}=p,re=O-1-p.padInfo.top,I=k-1-p.padInfo.left,ie=N/A;for(let e=0;e<D;++e)for(let t=0;t<A;++t)for(let n=0;n<j;++n){let r=n-re,i=Math.max(0,Math.ceil(r/F)),a=Math.min(ee,(O+r)/F);for(let o=0;o<M;++o){let s=o-I,c=Math.max(0,Math.ceil(s/ne)),l=Math.min(te,(k+s)/ne),u=0;for(let n=i;n<a;++n){let i=n*F-r;for(let r=c;r<l;++r){let a=r*ne-s,o=b*e+x*n+S*r,c=w*(O-1-i)+T*(k-1-a)+E*t;for(let e=0;e<ie;++e){let n=y[o+(t*ie+e)],r=C[c+e];u+=n*r}}}h[g*e+_*n+v*o+t]=u}}return n.makeTensorInfo(m.shape,m.dtype,m.values)}let Ud={kernelName:`DepthwiseConv2dNativeBackpropInput`,backendName:`cpu`,kernelFunc:Hd};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Wd(e){let{inputs:t,backend:n}=e,{x:r}=t,i=_(r.shape),a=n.data.get(r.dataId).values,o=W([i,i],r.dtype),s=o.values;for(let e=0;e<a.length;e++)s[e*i+e]=a[e];let c=[...r.shape,...r.shape];return n.makeTensorInfo(c,o.dtype,o.values)}let Gd={kernelName:$e,backendName:`cpu`,kernelFunc:Wd},Kd={kernelName:et,backendName:`cpu`,kernelFunc:({inputs:e,backend:t,attrs:n})=>{
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let{x:r,filter:i}=e,{strides:a,pad:o,dilations:s}=n,c=t,l=c.data.get(r.dataId).values,u=r.shape.length,d=c.data.get(i.dataId).values,f=i.shape.length,{batchSize:p,inHeight:m,inWidth:h,inChannels:g,outHeight:v,outWidth:y,padInfo:b,strideHeight:x,strideWidth:S,filterHeight:C,filterWidth:T,dilationHeight:E,dilationWidth:D,outShape:O}=Ca(r.shape,i.shape,a,o,`NHWC`,s),k=_(O),A=O.length,j=w(r.dtype,k);for(let e=0;e<p;++e)for(let t=0;t<v;++t){let n=t*x-b.top;for(let a=0;a<y;++a){let o=a*S-b.left;for(let s=0;s<g;++s){let c=-(2**53-1);for(let t=0;t<C;++t){let a=n+t*E;if(a>=0&&a<m)for(let n=0;n<T;++n){let p=o+n*D;if(p>=0&&p<h){let o=R([e,a,p,s],u,P(r.shape)),m=R([t,n,s],f,P(i.shape)),h=l[o]+d[m];h>c&&(c=h)}}}let p=R([e,t,a,s],A,P(O));j[p]=c}}}return{dataId:c.write(Cr(j,r.dtype),O,r.dtype),shape:O,dtype:r.dtype}}},qd={kernelName:nt,backendName:`cpu`,kernelFunc:({inputs:e,backend:t,attrs:n})=>{
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let{x:r,filter:i,dy:a}=e,{strides:o,pad:s,dilations:c}=n,l=t,u=F(r.shape,l.data.get(r.dataId).values),d=F(i.shape,l.data.get(i.dataId).values),{batchSize:f,inHeight:p,inWidth:h,inChannels:g,outHeight:_,outWidth:v,padInfo:y,strideHeight:b,strideWidth:x,filterHeight:S,filterWidth:C,dilationHeight:w,dilationWidth:T,outShape:E}=Ca(r.shape,i.shape,o,s,`NHWC`,c);m(a.rank===E.length,()=>`Error in ${nt}, dy must have the same rank as output ${E.length}, but got ${a.rank}`);let D=F(E,l.data.get(a.dataId).values),O=ie(i.shape,i.dtype);for(let e=0;e<f;++e)for(let t=0;t<_;++t){let n=t*b-y.top;for(let r=0;r<v;++r){let i=r*x-y.left;for(let a=0;a<g;++a){let o=-(2**53-1),s=0,c=0;for(let t=0;t<S;++t){let r=n+t*w;if(r>=0&&r<p)for(let n=0;n<C;++n){let l=i+n*T;if(l>=0&&l<h){let i=u[e][r][l][a]+d[t][n][a];i>o&&(o=i,s=t,c=n)}}}O[s][c][a]+=D[e][t][r][a]}}}return{dataId:l.write(Cr(O,r.dtype),i.shape,i.dtype),shape:i.shape,dtype:i.dtype}}},Jd={kernelName:tt,backendName:`cpu`,kernelFunc:({inputs:e,backend:t,attrs:n})=>{
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let{x:r,filter:i,dy:a}=e,{strides:o,pad:s,dilations:c}=n,l=t,u=F(r.shape,l.data.get(r.dataId).values),d=F(i.shape,l.data.get(i.dataId).values),{batchSize:f,inHeight:p,inWidth:h,inChannels:g,outHeight:_,outWidth:v,padInfo:y,strideHeight:b,strideWidth:x,filterHeight:S,filterWidth:C,dilationHeight:w,dilationWidth:T,outShape:E}=Ca(r.shape,i.shape,o,s,`NHWC`,c);m(a.rank===E.length,()=>`Error in ${tt}, dy must have the same rank as output ${E.length}, but got ${a.rank}`);let D=F(E,l.data.get(a.dataId).values),O=ie(r.shape,r.dtype);for(let e=0;e<f;++e)for(let t=0;t<_;++t){let n=t*b-y.top;for(let r=0;r<v;++r){let i=r*x-y.left;for(let a=0;a<g;++a){let o=-(2**53-1),s=n<0?0:n,c=i<0?0:i;for(let t=0;t<S;++t){let r=n+t*w;if(r>=0&&r<p)for(let n=0;n<C;++n){let l=i+n*T;if(l>=0&&l<h){let i=u[e][r][l][a]+d[t][n][a];i>o&&(o=i,s=r,c=l)}}}O[e][s][c][a]+=D[e][t][r][a]}}}return{dataId:l.write(Cr(O,r.dtype),r.shape,r.dtype),shape:r.shape,dtype:r.dtype}}};
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Yd(e){let{inputs:t,backend:n,attrs:r}=e,{image:i}=t,{canvas:a,options:o}=r,{contextOptions:s,imageOptions:c}=o||{},l=c?.alpha||1,u=s?.contextType||`2d`;if(u!==`2d`)throw Error(`Context type ${s.contextType} is not supported by the CPU backend.`);let d=a.getContext(u,s?.contextAttributes||{});if(d==null)throw Error(`Could not get the context with ${u} type.`);let[f,p]=i.shape.slice(0,2),m=i.shape.length===2?1:i.shape[2],h=n.data.get(i.dataId).values,g=i.dtype===`float32`?255:1,_=new Uint8ClampedArray(p*f*4);for(let e=0;e<f*p;++e){let t=[0,0,0,255*l];for(let n=0;n<m;n++){let r=h[e*m+n];if(i.dtype===`float32`){if(r<0||r>1)throw Error(`Tensor values for a float32 Tensor must be in the range [0 - 1] but encountered ${r}.`)}else if(i.dtype===`int32`&&(r<0||r>255))throw Error(`Tensor values for a int32 Tensor must be in the range [0 - 255] but encountered ${r}.`);m===1?(t[0]=r*g,t[1]=r*g,t[2]=r*g):t[n]=r*g}let n=e*4;_[n+0]=Math.round(t[0]),_[n+1]=Math.round(t[1]),_[n+2]=Math.round(t[2]),_[n+3]=Math.round(t[3])}a.width=p,a.height=f;let v=new ImageData(_,p,f);return d.putImageData(v,0,0),i}let Xd={kernelName:`Draw`,backendName:`cpu`,kernelFunc:Yd};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Zd(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{axis:a,keepDims:o}=r;q(i,`sum`);let s;s=i.dtype===`bool`?Oc({inputs:{x:i},backend:n,attrs:{dtype:`int32`}}):Cc({inputs:{x:i},backend:n});let c=s.shape.length,l=S(a,s.shape),u=uo(l,c),d=l,f=s;u!=null&&(f=ml({inputs:{x:s},backend:n,attrs:{perm:u}}),d=po(d.length,c)),lo(`sum`,d,f.shape.length);let[p,m]=so(f.shape,d),h=Sc(n,p,ti(f.dtype,`int32`)),g=_(m),v=n.data.get(h.dataId).values,y=n.data.get(f.dataId).values;for(let e=0;e<v.length;++e){let t=e*g,n=0;for(let e=0;e<g;++e)n+=y[t+e];v[e]=n}if(o){let e=co(h.shape,l),t=h;h=yu({inputs:{x:h},backend:n,attrs:{shape:e}}),n.disposeIntermediateTensorInfo(t)}return n.disposeIntermediateTensorInfo(s),u!=null&&n.disposeIntermediateTensorInfo(f),h}let Qd={kernelName:`Sum`,backendName:`cpu`,kernelFunc:Zd};
/**
* @license
* Copyright 2021 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function $d(e){let{inputs:t,backend:n,attrs:r}=e,{equation:i}=r,a=t,{allDims:o,summedDims:s,idDims:c}=qs(i,a.length);Ys(o.length,c,a);let{path:l,steps:u}=Xs(s,c),d=u.length,f=null,p=o.length,m=[];for(let e=0;e<d;++e){for(let t of u[e]){let{permutationIndices:e,expandDims:r}=Js(p,c[t]),i;Zs(e)?i=a[t]:(i=ml({inputs:{x:a[t]},backend:n,attrs:{perm:e}}),m.push(i));let o=i.shape.slice();for(let e=0;e<r.length;++e)o.splice(r[e],0,1);v(i.shape,o)||(i=yu({inputs:{x:i},backend:n,attrs:{shape:o}}),m.push(i)),f===null?f=i:(f=ol({inputs:{a:i,b:f},backend:n}),m.push(f))}e<d-1&&(l[e]>=0&&(f=Zd({inputs:{x:f},backend:n,attrs:{axis:l[e]-(o.length-p),keepDims:!1}}),m.push(f)),p--)}for(let e of m)e!==f&&n.disposeIntermediateTensorInfo(e);return f}let ef={kernelName:`Einsum`,backendName:`cpu`,kernelFunc:$d};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function tf(e){let{inputs:t,backend:n}=e,{dy:r,y:i}=t;q([r,i],`eluGrad`);let a=new Float32Array(_(i.shape)),o=n.data.get(i.dataId).values,s=n.data.get(r.dataId).values;for(let e=0;e<o.length;++e){let t=o[e];t>=0?a[e]=s[e]:a[e]=s[e]*(t+1)}return n.makeTensorInfo(i.shape,`float32`,a)}let nf={kernelName:it,backendName:`cpu`,kernelFunc:tf},rf={kernelName:`Erf`,backendName:`cpu`,kernelFunc:J(`Erf`,e=>{
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the License);
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an AS IS BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let t=Math.sign(e),n=Math.abs(e),r=1/(1+.3275911*n);return t*(1-((((1.061405429*r+-1.453152027)*r+1.421413741)*r+-.284496736)*r+.254829592)*r*Math.exp(-n*n))})};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function af(e){let{inputs:t,backend:n,attrs:r}=e,{input:i}=t,{dim:a}=r,o=i.shape.length,s=i.shape.slice(),c=a;return a<0&&(m(-(o+1)<=a,()=>`Axis must be in the interval [${-(o+1)}, ${o}]`),c=o+a+1),s.splice(c,0,1),yu({inputs:{x:i},backend:n,attrs:{shape:s}})}let of={kernelName:ot,backendName:`cpu`,kernelFunc:af},sf=Ac(rt,yc((e,t)=>e/t)),cf={kernelName:rt,backendName:`cpu`,kernelFunc:sf}
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function lf(e,t,n){let r=e.shape,i=r[0],a=r[1],o=n.data.get(e.dataId),s=o.complexTensorInfos.real,c=o.complexTensorInfos.imag,l=[i,a],u=_(l),d=C(`float32`,u),f=C(`float32`,u);for(let e=0;e<i;e++){let r=Bl({inputs:{x:s},backend:n,attrs:{begin:[e,0],size:[1,a]}}),i=Bl({inputs:{x:c},backend:n,attrs:{begin:[e,0],size:[1,a]}}),o=bc({inputs:{real:r,imag:i},backend:n}),{real:l,imag:u}=uf(o,t,n),p=Rs(l,u);for(let t=0;t<a;t++){let n=Hs(p,t);d[e*a+t]=n.real,f[e*a+t]=n.imag}n.disposeIntermediateTensorInfo(r),n.disposeIntermediateTensorInfo(i),n.disposeIntermediateTensorInfo(o)}let p=n.makeTensorInfo(l,`float32`,d),m=n.makeTensorInfo(l,`float32`,f),h=bc({inputs:{real:p,imag:m},backend:n});return n.disposeIntermediateTensorInfo(p),n.disposeIntermediateTensorInfo(m),h}function uf(e,t,n){let r=_(e.shape),i=n.data.get(e.dataId),a=n.data.get(i.complexTensorInfos.real.dataId).values,o=n.data.get(i.complexTensorInfos.imag.dataId).values;if(df(r)){let i=ff(a,o,r,t,n),s=[e.shape[0],e.shape[1]];if(t){let e=n.makeTensorInfo(s,`float32`,i.real),t=n.makeTensorInfo(s,`float32`,i.imag),a=n.makeTensorInfo([],`float32`,xr(r,`float32`)),o=Cc({inputs:{x:a},backend:n}),c=cf.kernelFunc({inputs:{a:e,b:a},backend:n}),l=cf.kernelFunc({inputs:{a:t,b:o},backend:n}),u=n.data.get(c.dataId).values,d=n.data.get(l.dataId).values;return n.disposeIntermediateTensorInfo(e),n.disposeIntermediateTensorInfo(t),n.disposeIntermediateTensorInfo(a),n.disposeIntermediateTensorInfo(o),n.disposeIntermediateTensorInfo(c),n.disposeIntermediateTensorInfo(l),{real:u,imag:d}}return i}return zs(pf(Rs(a,o),r,t))}function df(e){return!(e&e-1)}function ff(e,t,n,r,i){if(n===1)return{real:e,imag:t};let a=Rs(e,t),o=n/2,s=Bs(a),c=s.real,l=s.imag,u=[c.length],d=i.makeTensorInfo(u,`float32`,c),f=i.makeTensorInfo(u,`float32`,l),p=bc({inputs:{real:d,imag:f},backend:i}),m=Vs(a),h=m.real,g=m.imag,_=[h.length],v=i.makeTensorInfo(_,`float32`,h),y=i.makeTensorInfo(_,`float32`,g),b=bc({inputs:{real:v,imag:y},backend:i}),x=ff(c,l,o,r,i),S=x.real,C=x.imag,w=[S.length],T=i.makeTensorInfo(w,`float32`,S),E=i.makeTensorInfo(w,`float32`,C),D=bc({inputs:{real:T,imag:E},backend:i}),O=ff(h,g,o,r,i),k=O.real,A=O.imag,j=[k.length],M=i.makeTensorInfo(j,`float32`,k),N=i.makeTensorInfo(j,`float32`,A),ee=bc({inputs:{real:M,imag:N},backend:i}),P=Ws(n,r),te=[P.real.length],F=i.makeTensorInfo(te,`float32`,P.real),ne=i.makeTensorInfo(te,`float32`,P.imag),re=bc({inputs:{real:F,imag:ne},backend:i}),I=ol({inputs:{a:re,b:ee},backend:i}),ie=Mc({inputs:{a:D,b:I},backend:i}),L=eu({inputs:{a:D,b:I},backend:i}),R=Tc({inputs:{input:ie},backend:i}),ae=Tc({inputs:{input:L},backend:i}),oe=ud({inputs:{input:ie},backend:i}),se=ud({inputs:{input:L},backend:i}),ce=fd({inputs:[R,ae],backend:i,attrs:{axis:0}}),le=fd({inputs:[oe,se],backend:i,attrs:{axis:0}}),ue=i.data.get(ce.dataId).values,de=i.data.get(le.dataId).values;return i.disposeIntermediateTensorInfo(d),i.disposeIntermediateTensorInfo(f),i.disposeIntermediateTensorInfo(p),i.disposeIntermediateTensorInfo(v),i.disposeIntermediateTensorInfo(y),i.disposeIntermediateTensorInfo(b),i.disposeIntermediateTensorInfo(T),i.disposeIntermediateTensorInfo(E),i.disposeIntermediateTensorInfo(D),i.disposeIntermediateTensorInfo(M),i.disposeIntermediateTensorInfo(N),i.disposeIntermediateTensorInfo(ee),i.disposeIntermediateTensorInfo(F),i.disposeIntermediateTensorInfo(ne),i.disposeIntermediateTensorInfo(re),i.disposeIntermediateTensorInfo(I),i.disposeIntermediateTensorInfo(ie),i.disposeIntermediateTensorInfo(L),i.disposeIntermediateTensorInfo(R),i.disposeIntermediateTensorInfo(oe),i.disposeIntermediateTensorInfo(ae),i.disposeIntermediateTensorInfo(se),i.disposeIntermediateTensorInfo(ce),i.disposeIntermediateTensorInfo(le),{real:ue,imag:de}}function pf(e,t,n){let r=new Float32Array(t*2);for(let i=0;i<t;i++){let a=0,o=0;for(let r=0;r<t;r++){let s=Gs(i*r,t,n),c=Hs(e,r);a+=c.real*s.real-c.imag*s.imag,o+=c.real*s.imag+c.imag*s.real}n&&(a/=t,o/=t),Us(r,a,o,i)}return r}
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function mf(e){let{inputs:t,backend:n}=e,{input:r}=t,i=_(r.shape),a=r.shape[r.shape.length-1],o=i/a,s=yu({inputs:{x:r},backend:n,attrs:{shape:[o,a]}}),c=lf(s,!1,n),l=yu({inputs:{x:c},backend:n,attrs:{shape:r.shape}});return n.disposeIntermediateTensorInfo(s),n.disposeIntermediateTensorInfo(c),l}let hf={kernelName:`FFT`,backendName:`cpu`,kernelFunc:mf};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function gf(e){let{backend:t,attrs:n}=e,{shape:r,value:i,dtype:a}=n,o=a||N(i),s=w(o,_(r));return vf(s,i,o),t.makeTensorInfo(r,o,s)}let _f={kernelName:ct,backendName:`cpu`,kernelFunc:gf};function vf(e,t,n){e.fill(t)}
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let yf={kernelName:lt,backendName:`cpu`,kernelFunc:({inputs:e,attrs:t,backend:n})=>{let{image:r}=e,i=n,a=C(r.dtype,_(r.shape)),[o,s,c,l]=r.shape,u=i.data.get(r.dataId).values;for(let e=0;e<o;e++){let t=e*c*s*l;for(let e=0;e<s;e++){let n=c*l*e;for(let e=0;e<c;e++){let r=e*l;for(let i=0;i<l;i++){let o=Math.round(c-e-1),s=t+n+r+i,d=u[s];if(o>=0&&o<c){let e=o*l;d=u[t+n+e+i]}a[s]=d}}}}return{dataId:i.write(a,r.shape,r.dtype),shape:r.shape,dtype:r.dtype}}};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function bf(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,filter:a,bias:o,preluActivationWeights:s}=t,{strides:c,pad:l,dataFormat:u,dilations:d,dimRoundingMode:f,activation:p,leakyreluAlpha:m}=r,h=md({inputs:{x:i,filter:a},backend:n,attrs:{strides:c,pad:l,dataFormat:u,dilations:d,dimRoundingMode:f}});if(o){let e=h;if(u===`NCHW`&&o.shape.length===1&&o.shape[0]!==1){let e=yu({inputs:{x:o},backend:n,attrs:{shape:[o.shape[0],1,1]}});h=Mc({inputs:{a:h,b:e},backend:n}),n.disposeIntermediateTensorInfo(e)}else h=Mc({inputs:{a:h,b:o},backend:n});n.disposeIntermediateTensorInfo(e)}if(p){let e=h;if(u===`NCHW`&&p===`prelu`&&s.shape.length===1&&s.shape[0]!==1){let e=yu({inputs:{x:s},backend:n,attrs:{shape:[s.shape[0],1,1]}});h=vu(n,h,p,e,m),n.disposeIntermediateTensorInfo(e)}else h=vu(n,h,p,s,m);n.disposeIntermediateTensorInfo(e)}return h}let xf={kernelName:qn,backendName:`cpu`,kernelFunc:bf};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Sf(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,filter:a,bias:o,preluActivationWeights:s}=t,{strides:c,pad:l,dataFormat:u,dilations:d,dimRoundingMode:f,activation:p,leakyreluAlpha:m}=r,h=Rd({inputs:{x:i,filter:a},backend:n,attrs:{strides:c,pad:l,dataFormat:u,dilations:d,dimRoundingMode:f}});if(o){let e=h;h=Mc({inputs:{a:h,b:o},backend:n}),n.disposeIntermediateTensorInfo(e)}if(p){let e=h;h=vu(n,h,p,s,m),n.disposeIntermediateTensorInfo(e)}return h}let Cf={kernelName:Jn,backendName:`cpu`,kernelFunc:Sf};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function wf(e){let{inputs:t,backend:n}=e,{params:r,indices:i}=t,a=_(r.shape),o=i.shape,s=o[o.length-1],[c,l,u,d]=gs(r,i);if(l===0)return n.makeTensorInfo(c,r.dtype,[]);let f=n.data.get(i.dataId).values,p=Jc(f,n.bufferSync(r),r.dtype,l,s,u,d,r.shape,a);return n.makeTensorInfo(c,r.dtype,p.values)}let Tf={kernelName:mt,backendName:`cpu`,kernelFunc:wf};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Ef(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,indices:a}=t,{axis:o,batchDims:s}=r;q([i,a],`gatherV2`);let c=S(o,i.shape)[0],l=n.data.get(a.dataId).values,u=i.shape[c];for(let e=0;e<l.length;++e){let t=l[e];m(t<=u-1&&t>=0,()=>`GatherV2: the index value ${t} is not in [0, ${u-1}]`)}let d=s;s??(d=0);let f=_(a.shape),p=fc(i,a,c,d),h=yu({inputs:{x:i},backend:n,attrs:{shape:[p.batchSize,p.outerSize,p.dimSize,p.sliceSize]}}),g=yu({inputs:{x:a},backend:n,attrs:{shape:[p.batchSize,f/p.batchSize]}}),v=[p.batchSize,p.outerSize,f/p.batchSize,p.sliceSize],y=n.bufferSync(g),b=Yc(n.bufferSync(h),y,v);return n.disposeIntermediateTensorInfo(h),n.disposeIntermediateTensorInfo(g),n.makeTensorInfo(p.outputShape,b.dtype,b.values)}let Df={kernelName:pt,backendName:`cpu`,kernelFunc:Ef};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Of(e){let{inputs:t,backend:n}=e,{input:r}=t,i=_(r.shape),a=r.shape[r.shape.length-1],o=i/a,s=yu({inputs:{x:r},backend:n,attrs:{shape:[o,a]}}),c=lf(s,!0,n),l=yu({inputs:{x:c},backend:n,attrs:{shape:r.shape}});return n.disposeIntermediateTensorInfo(s),n.disposeIntermediateTensorInfo(c),l}let kf={kernelName:`IFFT`,backendName:`cpu`,kernelFunc:Of},Af={kernelName:vt,backendName:`cpu`,kernelFunc:J(vt,e=>+!!Number.isFinite(e),`bool`)},jf={kernelName:yt,backendName:`cpu`,kernelFunc:J(yt,e=>+(Math.abs(e)===1/0),`bool`)},Mf={kernelName:bt,backendName:`cpu`,kernelFunc:J(bt,e=>+!!Number.isNaN(e),`bool`)}
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the License);
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an AS IS BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the License);
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an AS IS BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the License);
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an AS IS BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Nf(e){let{backend:t,attrs:n}=e,{start:r,stop:i,num:a}=n,o=el(r,i,a);return t.makeTensorInfo([o.length],`float32`,o)}let Pf={kernelName:wt,backendName:`cpu`,kernelFunc:Nf},Ff={kernelName:Tt,backendName:`cpu`,kernelFunc:J(Tt,e=>Math.log1p(e))},If={kernelName:Et,backendName:`cpu`,kernelFunc:Ac(Et,yc((e,t)=>e&&t),null,`bool`)},Lf={kernelName:Dt,backendName:`cpu`,kernelFunc:J(Dt,e=>+!e,`bool`)},Rf={kernelName:Ot,backendName:`cpu`,kernelFunc:Ac(Ot,yc((e,t)=>e||t),null,`bool`)}
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the License);
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an AS IS BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the License);
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an AS IS BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function zf(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{depthRadius:a,bias:o,alpha:s,beta:c}=r;q(i,`LRN`);let l=i.shape[3],u=l-1,d=n.data.get(i.dataId).values,f=_(i.shape),p=new Float32Array(f);function m(e){let t=e%l,n=e-t+Math.max(0,t-a),r=e-t+Math.min(t+a,u),i=0;for(;n<=r;n++){let e=d[n];i+=e*e}return i}for(let e=0;e<f;e++){let t=m(e),n=d[e]*(o+s*t)**+-c;p[e]=n}return n.makeTensorInfo(i.shape,i.dtype,p)}let Bf={kernelName:`LRN`,backendName:`cpu`,kernelFunc:zf};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Vf(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,y:a,dy:o}=t,{depthRadius:s,bias:c,alpha:l,beta:u}=r;q(o,`LRNGrad`);let d=_(o.shape),f=o.shape[3],p=n.data.get(o.dataId).values,m=n.data.get(i.dataId).values,h=n.data.get(a.dataId).values,g=new Float32Array(d),v=d;for(let e=0;e<v;e++){let t=e%f,n=e-t+Math.max(0,t-s),r=e-t+Math.min(f,t+s+1),i=0;for(let e=n;e<r;e++)i+=m[e]**2;i=l*i+c;for(let t=n;t<r;t++){let n=-2*l*u*m[t]*h[e]/i;e===t&&(n+=i**+-u),n*=p[e],g[t]+=n}}return n.makeTensorInfo(o.shape,i.dtype,g)}let Hf={kernelName:kt,backendName:`cpu`,kernelFunc:Vf};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Uf(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{reductionIndices:a,keepDims:o}=r,s=n,c=i.shape,l=c.length,u=S(a,c),d=u,f=uo(d,l),p=s.data.get(i.dataId).values;if(f!=null){let e=Array(l);for(let t=0;t<e.length;t++)e[t]=c[f[t]];p=pl(p,c,i.dtype,f,e),d=po(d.length,l),c=e}q(i,`max`),lo(`max`,d,l);let[m,h]=so(c,d),g=_(h),v=nl(p,g,m,i.dtype),y=s.write(v,m,i.dtype),b=m;return o&&(b=co(m,u)),{dataId:y,shape:b,dtype:i.dtype}}let Wf={kernelName:`Max`,backendName:`cpu`,kernelFunc:Uf};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Gf(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t;q(i,`maxPool`);let{filterSize:a,strides:o,pad:s,dimRoundingMode:c}=r;m(Ra(o,1),()=>`Error in maxPool: Either strides or dilations must be 1. Got strides ${o} and dilations '1'`);let l=wa(i.shape,a,o,1,s,c),u;if(l.filterWidth===1&&l.filterHeight===1&&v(l.inShape,l.outShape))u=Cc({inputs:{x:i},backend:n});else{let e=n.data.get(i.dataId).values,t=P(i.shape),r=Hu(e,i.shape,i.dtype,t,l,`max`);u=n.makeTensorInfo(l.outShape,i.dtype,r.values)}return u}let Kf={kernelName:jt,backendName:`cpu`,kernelFunc:Gf};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function qf(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{filterSize:a,strides:o,pad:s,dimRoundingMode:c,dataFormat:l}=r;q(i,`maxPool3d`);let u=Ta(i.shape,a,o,1,s,c,l),d=n.data.get(i.dataId).values,f=Wu(d,i.shape,i.dtype,P(i.shape),u,`max`);return n.makeTensorInfo(f.shape,`float32`,f.values)}let Jf={kernelName:Nt,backendName:`cpu`,kernelFunc:qf};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Yf(e){let{inputs:t,backend:n,attrs:r}=e,{dy:i,input:a}=t,{filterSize:o,strides:s,pad:c,dimRoundingMode:l}=r;q([i,a],`maxPool3DGrad`);let u=Ta(a.shape,o,s,1,c,l),d=Gu(n.bufferSync(a),u),f=u.strideDepth,p=u.strideHeight,m=u.strideWidth,h=u.dilationDepth,g=u.dilationHeight,_=u.dilationWidth,v=u.effectiveFilterDepth,y=u.effectiveFilterHeight,b=u.effectiveFilterWidth,x=v-1-u.padInfo.front,S=b-1-u.padInfo.left,C=y-1-u.padInfo.top,w=W(a.shape,`float32`),T=n.bufferSync(i);for(let e=0;e<u.batchSize;++e)for(let t=0;t<u.inChannels;++t)for(let n=0;n<u.inDepth;++n)for(let r=0;r<u.inHeight;++r)for(let i=0;i<u.inWidth;++i){let a=n-x,o=r-C,s=i-S,c=0;for(let n=0;n<v;n+=h){let r=(a+n)/f;if(!(r<0||r>=u.outDepth||Math.floor(r)!==r))for(let i=0;i<y;i+=g){let a=(o+i)/p;if(!(a<0||a>=u.outHeight||Math.floor(a)!==a))for(let o=0;o<b;o+=_){let l=(s+o)/m;if(l<0||l>=u.outWidth||Math.floor(l)!==l)continue;let f=+(v*y*b-1-d.get(e,r,a,l,t)===n*y*b+i*b+o);if(f===0)continue;let p=T.get(e,r,a,l,t);c+=p*f}}}w.set(c,e,n,r,i,t)}return n.makeTensorInfo(w.shape,w.dtype,w.values)}let Xf={kernelName:Pt,backendName:`cpu`,kernelFunc:Yf};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Zf(e){let{inputs:t,backend:n,attrs:r}=e,{dy:i,input:a,output:o}=t,s=a;q([a,o],`maxPoolGrad`);let{filterSize:c,strides:l,pad:u,dimRoundingMode:d}=r,f=wa(s.shape,c,l,1,u,d),p=n.data.get(s.dataId).values,m=W(f.outShape,s.dtype,Uu(p,s.shape,s.dtype,f).values),h=f.strideHeight,g=f.strideWidth,_=f.dilationHeight,v=f.dilationWidth,y=f.effectiveFilterHeight,b=f.effectiveFilterWidth,x=b-1-f.padInfo.left,S=y-1-f.padInfo.top,C=W(s.shape,`float32`),w=n.data.get(i.dataId).values,T=W(i.shape,`float32`,w);for(let e=0;e<f.batchSize;++e)for(let t=0;t<f.inChannels;++t)for(let n=0;n<f.inHeight;++n)for(let r=0;r<f.inWidth;++r){let i=n-S,a=r-x,o=0;for(let n=0;n<y;n+=_){let r=(i+n)/h;if(!(r<0||r>=f.outHeight||Math.floor(r)!==r))for(let i=0;i<b;i+=v){let s=(a+i)/g;if(s<0||s>=f.outWidth||Math.floor(s)!==s)continue;let c=+(y*b-1-m.get(e,r,s,t)===n*b+i);if(c===0)continue;let l=T.get(e,r,s,t);o+=l*c}}C.set(o,e,n,r,t)}return n.makeTensorInfo(C.shape,C.dtype,C.values)}let Qf={kernelName:Mt,backendName:`cpu`,kernelFunc:Zf};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function $f(e,t,n,r,i){let a=Hu(e,t,n,P(t),i,`max`),o=Uu(e,t,n,i,!0,r);return[a.values,o.values]}
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let ep={kernelName:Ft,backendName:`cpu`,kernelFunc:({inputs:e,attrs:t,backend:n})=>{let{x:r}=e,{filterSize:i,strides:a,pad:o,includeBatchInIndex:s}=t,c=n;q(r,`MaxPoolWithArgmax`);let l=c.data.get(r.dataId).values,u=wa(r.shape,i,a,[1,1],o),[d,f]=$f(l,r.shape,r.dtype,s,u),p=c.write(d,u.outShape,r.dtype),m=c.write(f,u.outShape,r.dtype);return[{dataId:p,shape:u.outShape,dtype:r.dtype},{dataId:m,shape:u.outShape,dtype:`int32`}]}};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function tp(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{axis:a,keepDims:o}=r,s=S(a,i.shape),c=so(i.shape,s)[1],l=_(c),u=[],d=n.makeTensorInfo([],`float32`,new Float32Array([l]));u.push(d);let f=Oc({inputs:{x:i},backend:n,attrs:{dtype:`float32`}});u.push(f);let p=sf({inputs:{a:f,b:d},backend:n});u.push(p);let m=Zd({inputs:{x:p},backend:n,attrs:{axis:a,keepDims:o}});return u.forEach(e=>n.disposeIntermediateTensorInfo(e)),m}let np={kernelName:It,backendName:`cpu`,kernelFunc:tp};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function rp(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{axis:a,keepDims:o}=r;q(i,`min`);let s=S(a,i.shape),c=s,l=uo(c,i.shape.length),u=i;l!=null&&(u=ml({inputs:{x:i},backend:n,attrs:{perm:l}}),c=po(c.length,i.shape.length)),lo(`min`,c,u.shape.length);let[d,f]=so(u.shape,c),p=_(f),m=I(_(d),u.dtype),h=n.data.get(u.dataId).values;for(let e=0;e<m.length;++e){let t=e*p,n=h[t];for(let e=0;e<p;++e){let r=h[t+e];(Number.isNaN(r)||r<n)&&(n=r)}m[e]=n}l!=null&&n.disposeIntermediateTensorInfo(u);let g=n.makeTensorInfo(d,u.dtype,m);if(o){let e=co(d,s),t=yu({inputs:{x:g},backend:n,attrs:{shape:e}});return n.disposeIntermediateTensorInfo(g),t}return g}let ip={kernelName:`Min`,backendName:`cpu`,kernelFunc:rp};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function ap(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{paddings:a,mode:o}=r;q(i,`mirrorPad`);let s=a.map((e,t)=>e[0]+i.shape[t]+e[1]),c=a.map(e=>e[0]),l=a.map((e,t)=>e[0]+i.shape[t]),u=o===`reflect`?0:1,d=n.data.get(i.dataId).values,f=i.shape.length,p=P(i.shape),m=_(s),h=s.length,g=P(s),v=C(i.dtype,m);for(let e=0;e<m;e++){let t=ae(e,h,g);for(let e=0;e<h;e++)t[e]<c[e]?t[e]=c[e]*2-t[e]-u:t[e]>=l[e]&&(t[e]=(l[e]-1)*2-t[e]+u);t=t.map((e,t)=>e-c[t]);let n=R(t,f,p);v[e]=d[n]}return{dataId:n.write(v,s,i.dtype),shape:s,dtype:i.dtype}}let op={kernelName:Rt,backendName:`cpu`,kernelFunc:ap},sp={kernelName:`Mod`,backendName:`cpu`,kernelFunc:Ac(`Mod`,yc(((e,t)=>{
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let n=e%t;return e<0&&t<0||e>=0&&t>=0?n:(n+t)%t})))};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function cp(e){let{inputs:t,backend:n,attrs:r}=e,{logits:i}=t,{dim:a}=r,o=i.shape.length,s=a;if(s===-1&&(s=o-1),s!==o-1)throw Error(`Softmax along a non-last dimension is not yet supported. Logits was rank ${o} and dim was ${s}`);let c=S([s],i.shape),l=Uf({inputs:{x:i},backend:n,attrs:{reductionIndices:c,keepDims:!1}}),u=co(l.shape,c),d=yu({inputs:{x:l},backend:n,attrs:{shape:u}}),f=eu({inputs:{a:i,b:d},backend:n}),p=Uc({inputs:{x:f},backend:n}),m=Zd({inputs:{x:p},backend:n,attrs:{axis:c,keepDims:!1}}),h=yu({inputs:{x:m},backend:n,attrs:{shape:u}}),g=sf({inputs:{a:p,b:h},backend:n});return n.disposeIntermediateTensorInfo(l),n.disposeIntermediateTensorInfo(d),n.disposeIntermediateTensorInfo(f),n.disposeIntermediateTensorInfo(p),n.disposeIntermediateTensorInfo(m),n.disposeIntermediateTensorInfo(h),g}let lp={kernelName:Cn,backendName:`cpu`,kernelFunc:cp};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function up(e){let{inputs:t,backend:n,attrs:r}=e,{logits:i}=t,{numSamples:a,seed:o,normalized:s}=r;q(i,`multinomial`);let c=s?i:cp({inputs:{logits:i},backend:n,attrs:{dim:-1}}),l=c.shape[0],u=c.shape[1],d=n.data.get(c.dataId).values,f=[l,a],p=I(_(f),`int32`);for(let e=0;e<l;++e){let t=e*u,n=new Float32Array(u-1);n[0]=d[t];for(let e=1;e<n.length;++e)n[e]=n[e-1]+d[t+e];let r=cl.alea(o.toString()),i=e*a;for(let e=0;e<a;++e){let t=r();p[i+e]=n.length;for(let r=0;r<n.length;r++)if(t<n[r]){p[i+e]=r;break}}}return s||n.disposeIntermediateTensorInfo(c),n.makeTensorInfo(f,`int32`,p)}let dp={kernelName:zt,backendName:`cpu`,kernelFunc:up},fp=Xo
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;function pp(e){let{inputs:t,backend:n,attrs:r}=e,{boxes:i,scores:a}=t,{maxOutputSize:o,iouThreshold:s,scoreThreshold:c}=r;q(i,`NonMaxSuppression`);let l=n.data.get(i.dataId).values,u=n.data.get(a.dataId).values,{selectedIndices:d}=fp(l,u,o,s,c);return n.makeTensorInfo([d.length],`int32`,new Int32Array(d))}let mp={kernelName:Ht,backendName:`cpu`,kernelFunc:pp},hp=Zo
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;function gp(e){let{inputs:t,backend:n,attrs:r}=e,{boxes:i,scores:a}=t,{maxOutputSize:o,iouThreshold:s,scoreThreshold:c,padToMaxOutputSize:l}=r;q(i,`NonMaxSuppressionPadded`);let u=n.data.get(i.dataId).values,d=n.data.get(a.dataId).values,{selectedIndices:f,validOutputs:p}=hp(u,d,o,s,c,l);return[n.makeTensorInfo([f.length],`int32`,new Int32Array(f)),n.makeTensorInfo([],`int32`,new Int32Array([p]))]}let _p={kernelName:Ut,backendName:`cpu`,kernelFunc:gp},vp=Qo
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;function yp(e){let{inputs:t,backend:n,attrs:r}=e,{boxes:i,scores:a}=t,{maxOutputSize:o,iouThreshold:s,scoreThreshold:c,softNmsSigma:l}=r;q(i,`NonMaxSuppressionWithScore`);let u=n.data.get(i.dataId).values,d=n.data.get(a.dataId).values,{selectedIndices:f,selectedScores:p}=vp(u,d,o,s,c,l);return[n.makeTensorInfo([f.length],`int32`,new Int32Array(f)),n.makeTensorInfo([p.length],`float32`,new Float32Array(p))]}let bp={kernelName:Wt,backendName:`cpu`,kernelFunc:yp};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function xp(e){let{inputs:t,backend:n,attrs:r}=e,{indices:i}=t,{dtype:a,depth:o,onValue:s,offValue:c}=r;q(i,`oneHot`);let l=_(i.shape),u=new Float32Array(l*o);u.fill(c);let d=n.data.get(i.dataId).values;for(let e=0;e<l;++e)d[e]>=0&&d[e]<o&&(u[e*o+d[e]]=s);return n.makeTensorInfo([...i.shape,o],a,u)}let Sp={kernelName:Kt,backendName:`cpu`,kernelFunc:xp};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Cp(e){let{inputs:t,backend:n}=e,{x:r}=t;if(r.dtype===`string`)throw Error(`zerosLike is not supported for string tensors`);if(r.dtype===`complex64`){let e=Tc({inputs:{input:r},backend:n}),t=Cp({inputs:{x:e},backend:n}),i=ud({inputs:{input:r},backend:n}),a=Cp({inputs:{x:i},backend:n}),o=bc({inputs:{real:t,imag:a},backend:n});return n.disposeIntermediateTensorInfo(e),n.disposeIntermediateTensorInfo(t),n.disposeIntermediateTensorInfo(i),n.disposeIntermediateTensorInfo(a),o}return gf({backend:n,attrs:{shape:r.shape,value:0,dtype:r.dtype}})}let wp={kernelName:Un,backendName:`cpu`,kernelFunc:Cp};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Tp(e){let{inputs:t,backend:n}=e,{x:r}=t;if(r.dtype===`string`)throw Error(`onesLike is not supported for string tensors`);if(r.dtype===`complex64`){let e=Tc({inputs:{input:r},backend:n}),t=Tp({inputs:{x:e},backend:n}),i=ud({inputs:{input:r},backend:n}),a=Cp({inputs:{x:i},backend:n}),o=bc({inputs:{real:t,imag:a},backend:n});return n.disposeIntermediateTensorInfo(e),n.disposeIntermediateTensorInfo(t),n.disposeIntermediateTensorInfo(i),n.disposeIntermediateTensorInfo(a),o}return gf({backend:n,attrs:{shape:r.shape,value:1,dtype:r.dtype}})}let Ep={kernelName:Gt,backendName:`cpu`,kernelFunc:Tp};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Dp(e){let{inputs:t,backend:n,attrs:r}=e,{axis:i}=r;if(t.length===1)return af({inputs:{input:t[0]},backend:n,attrs:{dim:i}});let a=t[0].shape,o=t[0].dtype;t.forEach(e=>{h(a,e.shape,`All tensors passed to stack must have matching shapes`),m(o===e.dtype,()=>`All tensors passed to stack must have matching dtypes`)});let s=[],c=fd({inputs:t.map(e=>{let t=af({inputs:{input:e},backend:n,attrs:{dim:i}});return s.push(t),t}),backend:n,attrs:{axis:i}});return s.forEach(e=>n.disposeIntermediateTensorInfo(e)),c}let Op={kernelName:qt,backendName:`cpu`,kernelFunc:Dp};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function kp(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{paddings:a,constantValue:o}=r;q(i,`pad`);let s=a.map((e,t)=>e[0]+i.shape[t]+e[1]),c=a.map(e=>e[0]),l=n.data.get(i.dataId).values,u=_(i.shape),d=i.shape.length,f=P(i.shape),p=_(s),m=s.length,h=P(s),g=C(i.dtype,p);o!==0&&g.fill(o);for(let e=0;e<u;e++){let t=R(ae(e,d,f).map((e,t)=>e+c[t]),m,h);g[t]=l[e]}return{dataId:n.write(g,s,i.dtype),shape:s,dtype:i.dtype}}let Ap={kernelName:Jt,backendName:`cpu`,kernelFunc:kp},jp={kernelName:`Pow`,backendName:`cpu`,kernelFunc:Ac(`Pow`,yc((e,t)=>e**+t))}
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;
/**
* @license
* Copyright 2022 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Mp(e){let{inputs:t,backend:n,attrs:r}=e,{paramsNestedSplits:i,paramsDenseValues:a,indices:o}=t,{outputRaggedRank:s}=r,c=i.map(e=>n.data.get(e.dataId).values),l=i.map(e=>e.shape),u=n.data.get(a.dataId).values,d=n.data.get(o.dataId).values,[f,p,m]=El(c,l,u,a.shape,a.dtype,d,o.shape,s),h=f.map(e=>n.makeTensorInfo([e.length],`int32`,e)),g=n.makeTensorInfo(m,a.dtype,p);return h.concat([g])}let Np={kernelName:`RaggedGather`,backendName:`cpu`,kernelFunc:Mp};
/**
* @license
* Copyright 2022 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Pp(e){let{inputs:t,backend:n}=e,{starts:r,limits:i,deltas:a}=t,o=n.data.get(r.dataId).values,s=n.data.get(i.dataId).values,c=n.data.get(a.dataId).values,[l,u]=Ol(o,r.shape,r.dtype,s,i.shape,c,a.shape);return[n.makeTensorInfo([l.length],`int32`,l),n.makeTensorInfo([u.length],r.dtype,u)]}let Fp={kernelName:`RaggedRange`,backendName:`cpu`,kernelFunc:Pp};
/**
* @license
* Copyright 2022 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Ip(e){let{inputs:t,backend:n,attrs:r}=e,{shape:i,values:a,defaultValue:o,rowPartitionTensors:s}=t,{rowPartitionTypes:c}=r,l=n.data.get(i.dataId).values,u=n.data.get(a.dataId).values,d=n.data.get(o.dataId).values,f=s.map(e=>n.data.get(e.dataId).values),p=s.map(e=>e.shape),[m,h]=Nl(l,i.shape,u,a.shape,a.dtype,d,o.shape,f,p,c);return n.makeTensorInfo(m,a.dtype,h)}let Lp={kernelName:`RaggedTensorToTensor`,backendName:`cpu`,kernelFunc:Ip};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Rp(e){let{backend:t,attrs:n}=e,{start:r,stop:i,dtype:a,step:o}=n,s=Pl(r,i,o,a);return t.makeTensorInfo([s.length],a,s)}let zp={kernelName:Zt,backendName:`cpu`,kernelFunc:Rp},Bp={kernelName:Qt,backendName:`cpu`,kernelFunc:J(Qt,e=>1/e)}
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the License);
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an AS IS BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Vp(e){let{inputs:t,backend:n,attrs:r}=e,{images:i}=t,{alignCorners:a,halfPixelCenters:o,size:s}=r;q(i,`resizeBilinear`);let c=P(i.shape),[l,u]=s,[d,f,p,m]=i.shape,h=n.data.get(i.dataId).values,g=new Float32Array(_([d,l,u,m])),v=[a&&l>1?f-1:f,a&&u>1?p-1:p],y=[a&&l>1?l-1:l,a&&u>1?u-1:u],b=0,x=v[0]/y[0],S=v[1]/y[1];for(let e=0;e<d;e++)for(let t=0;t<l;t++){let n;n=o?x*(t+.5)-.5:x*t;let r=Math.max(0,Math.floor(n)),i=n-r,a=Math.min(f-1,Math.ceil(n)),s=e*c[0]+r*c[1],l=e*c[0]+a*c[1];for(let e=0;e<u;e++){let t;t=o?S*(e+.5)-.5:S*e;let n=Math.max(0,Math.floor(t)),r=t-n,a=Math.min(p-1,Math.ceil(t)),u=s+n*c[2],d=l+n*c[2],f=s+a*c[2],_=l+a*c[2];for(let e=0;e<m;e++){let t=h[u+e],n=h[d+e],a=h[f+e],o=h[_+e],s=t+(a-t)*r,c=s+(n+(o-n)*r-s)*i;g[b++]=c}}}return n.makeTensorInfo([d,l,u,m],`float32`,g)}let Hp={kernelName:rn,backendName:`cpu`,kernelFunc:Vp};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Up(e){let{inputs:t,backend:n,attrs:r}=e,{images:i,dy:a}=t,{alignCorners:o}=r;q([a,i],`resizeBilinearGrad`);let s=P(i.shape),[c,l,u,d]=i.shape,[,f,p]=a.shape,m=new Float32Array(c*l*u*d),h=[o&&f>1?l-1:l,o&&p>1?u-1:u],g=[o&&f>1?f-1:f,o&&p>1?p-1:p],_=h[0]/g[0],v=h[1]/g[1],y=n.data.get(a.dataId).values,b=0;for(let e=0;e<c;e++){let t=e*s[0];for(let e=0;e<f;e++){let n=e*_,r=Math.floor(n),i=Math.min(Math.ceil(n),l-1),a=t+r*s[1],o=t+i*s[1],c=n-r,f=1-c;for(let e=0;e<p;e++){let t=e*v,n=Math.floor(t),r=Math.min(Math.ceil(t),u-1),i=t-n,l=1-i,p=a+n*s[2],h=a+r*s[2],g=o+n*s[2],_=o+r*s[2],x=f*l,S=f*i,C=c*l,w=c*i;for(let e=0;e<d;e++){let t=y[b++];m[p+e]+=t*x,m[h+e]+=t*S,m[g+e]+=t*C,m[_+e]+=t*w}}}}return n.makeTensorInfo([c,u,l,d],`float32`,m)}let Wp={kernelName:an,backendName:`cpu`,kernelFunc:Up};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Gp(e){let{inputs:t,backend:n,attrs:r}=e,{images:i}=t,{alignCorners:a,halfPixelCenters:o,size:s}=r;q(i,`resizeNearestNeighbor`);let c=P(i.shape),[l,u]=s,[d,f,p,m]=i.shape,h=n.data.get(i.dataId).values,g=new Float32Array(d*l*u*m),_=[a&&l>1?f-1:f,a&&u>1?p-1:p],v=[a&&l>1?l-1:l,a&&u>1?u-1:u],y=_[0]/v[0],b=_[1]/v[1],x=0;for(let e=0;e<d;e++){let t=e*c[0];for(let e=0;e<l;e++){let n=o?y*(e+.5):y*e,r=Math.min(f-1,a?Math.round(n):Math.floor(n));o&&(r=Math.max(0,r));let i=t+r*c[1];for(let e=0;e<u;e++){let t=o?b*(e+.5):b*e,n=Math.min(p-1,a?Math.round(t):Math.floor(t));o&&(n=Math.max(0,n));let r=i+n*c[2];for(let e=0;e<m;e++){let t=h[r+e];g[x++]=t}}}}return n.makeTensorInfo([d,l,u,m],i.dtype,g)}let Kp={kernelName:tn,backendName:`cpu`,kernelFunc:Gp};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function qp(e){let{inputs:t,backend:n,attrs:r}=e,{images:i,dy:a}=t,{alignCorners:o}=r;q([a,i],`resizeNearestNeighborGrad`);let s=P(i.shape),c=P(a.shape),[l,u,d,f]=i.shape,[,p,m]=a.shape,h=new Float32Array(l*u*d*f),g=n.data.get(a.dataId).values,_=[o&&p>1?u-1:u,o&&m>1?d-1:d],v=[o&&p>1?p-1:p,o&&m>1?m-1:m],y=_[0]/v[0],b=_[1]/v[1],x=1/y,S=1/b,C=Math.ceil(x)*2+2,w=Math.ceil(S)*2+2;for(let e=0;e<l;e++){let t=e*s[0];for(let e=0;e<u;e++){let n=t+e*s[1],r=Math.floor(e*x),i=Math.floor(r-C/2);for(let r=0;r<d;r++){let a=n+r*s[2],l=Math.floor(r*S),_=Math.floor(l-w/2);for(let n=0;n<f;n++){let s=0;for(let a=0;a<C;a++){let l=a+i;if(l<0||l>=p)continue;let f=t+l*c[1],h=l*y,v=Math.min(u-1,o?Math.round(h):Math.floor(h));if(e===v)for(let e=0;e<w;e++){let t=e+_;if(t<0||t>=m)continue;let i=f+t*c[2],a=t*b,l=Math.min(d-1,o?Math.round(a):Math.floor(a));r===l&&(s+=g[i+n])}}h[a+n]=s}}}}return n.makeTensorInfo(i.shape,i.dtype,h)}let Jp={kernelName:nn,backendName:`cpu`,kernelFunc:qp};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Yp(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{dims:a}=r;q(i,`reverse`);let o=i.shape.length,s=S(a,i.shape);if(o===0)return Cc({inputs:{x:i},backend:n});let c=new Vr(i.shape,i.dtype),l=n.bufferSync(i);for(let e=0;e<c.size;e++){let t=c.indexToLoc(e),n=t.slice();s.forEach(e=>n[e]=i.shape[e]-1-n[e]),c.set(l.get(...n),...t)}return n.makeTensorInfo(c.shape,c.dtype,c.values)}let Xp={kernelName:sn,backendName:`cpu`,kernelFunc:Yp},Zp={kernelName:Gn,backendName:`cpu`,kernelFunc:({inputs:e,attrs:t,backend:n})=>{
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let{image:r}=e,{radians:i,fillValue:a,center:o}=t,s=n,c=C(r.dtype,_(r.shape)),[l,u,d,f]=r.shape,[p,m]=Ms(o,u,d),h=Math.sin(i),g=Math.cos(i),v=s.data.get(r.dataId).values;for(let e=0;e<l;e++){let t=e*d*u*f;for(let e=0;e<u;e++){let n=d*f*e;for(let r=0;r<d;r++){let i=r*f;for(let o=0;o<f;o++){let s=[l,e,r,o],_=s[2],y=s[1],b=(_-p)*g-(y-m)*h,x=(_-p)*h+(y-m)*g;b=Math.round(b+p),x=Math.round(x+m);let S=a;if(typeof a!=`number`&&(S=o===3?255:a[o]),b>=0&&b<d&&x>=0&&x<u){let e=d*f*x,n=b*f;S=v[t+e+n+o]}let C=t+n+i+o;c[C]=S}}}}return{dataId:s.write(c,r.shape,r.dtype),shape:r.shape,dtype:r.dtype}}},Qp={kernelName:cn,backendName:`cpu`,kernelFunc:J(cn,e=>{
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the License);
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an AS IS BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let t=Math.floor(e);return e-t<.5?Math.floor(e):e-t>.5?Math.ceil(e):t%2==0?t:t+1})};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function $p(e){let{inputs:t,backend:n,attrs:r}=e,{indices:i,updates:a}=t,{shape:o}=r,{sliceRank:s,numUpdates:c,sliceSize:l,strides:u,outputSize:d}=Wo(a,i,o),f=Il(n.bufferSync(i),n.bufferSync(a),o,d,l,c,s,u,0,!0);return n.makeTensorInfo(o,f.dtype,f.values)}let em={kernelName:un,backendName:`cpu`,kernelFunc:$p};
/**
* @license
* Copyright 2022 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function tm(e,t){let n=0,r=e.length,i=0;for(;n<r;)i=Math.floor((n+r)/2),e[i]<t?n=i+1:r=i;return r}function nm(e,t){let n=0,r=e.length,i=0;for(;n<r;)i=Math.floor((n+r)/2),e[i]<=t?n=i+1:r=i;return r}function rm(e,t,n,r,i,a){let o=w(`int32`,n*i);for(let s=0;s<n;++s){let n=e.slice(s*r,(s+1)*r),c=s*i;for(let e=0;e<i;++e)o[c+e]=a===`left`?tm(n,t[e+c]):nm(n,t[e+c])}return o}
/**
* @license
* Copyright 2022 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function im(e){let{inputs:t,backend:n,attrs:r}=e,{sortedSequence:i,values:a}=t,{side:o}=r,s=n.data.get(i.dataId).values,c=n.data.get(a.dataId).values,l=rm(s,c,i.shape[0],i.shape[1],a.shape[1],o);return n.makeTensorInfo(a.shape,`int32`,l)}let am={kernelName:fn,backendName:`cpu`,kernelFunc:im};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function om(e){let{inputs:t,backend:n}=e,{condition:r,t:i,e:a}=t;q([r,i,a],`select`);let o=r.shape.length,s=n.data.get(r.dataId).values,c=n.data.get(i.dataId).values,l=n.data.get(a.dataId).values,u=ti(i.dtype,a.dtype),d=I(_(i.shape),u),f=0,p=o===0||o>1||i.shape.length===1?1:_(i.shape.slice(1));for(let e=0;e<s.length;e++)for(let t=0;t<p;t++)s[e]===1?d[f++]=c[e]:d[f++]=l[e];return n.makeTensorInfo(i.shape,u,d)}let sm={kernelName:pn,backendName:`cpu`,kernelFunc:om},cm={kernelName:mn,backendName:`cpu`,kernelFunc:J(mn,e=>e>=0?1.0507009873554805*e:1.7580993408473768*(Math.exp(e)-1))},lm={kernelName:_n,backendName:`cpu`,kernelFunc:J(_n,e=>e<0?-1:+(e>0))},um={kernelName:`Sin`,backendName:`cpu`,kernelFunc:J(`Sin`,e=>Math.sin(e))},dm={kernelName:gn,backendName:`cpu`,kernelFunc:J(gn,e=>Math.sinh(e))},fm=Math.log(1.1920928955078125e-7)+2,pm={kernelName:yn,backendName:`cpu`,kernelFunc:J(yn,e=>{
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the License);
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an AS IS BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the License);
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an AS IS BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the License);
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an AS IS BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the License);
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an AS IS BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the License);
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an AS IS BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let t=e>-fm,n=e<fm,r=Math.exp(e),i;return i=n?r:t?e:Math.log(1+r),i})};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function mm(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{blockShape:a,paddings:o}=r;q([i],`spaceToBatchND`);let s=_(a),c=[[0,0]];c.push(...o);for(let e=1+a.length;e<i.shape.length;++e)c.push([0,0]);let l=Ap.kernelFunc({inputs:{x:i},backend:n,attrs:{paddings:c,constantValue:0}}),u=Ns(l.shape,a,s,!1),d=Ps(u.length,a.length,!1),f=Fs(l.shape,a,s,!1),p=yu({inputs:{x:l},backend:n,attrs:{shape:u}}),m=ml({inputs:{x:p},backend:n,attrs:{perm:d}}),h=yu({inputs:{x:m},backend:n,attrs:{shape:f}});return n.disposeIntermediateTensorInfo(l),n.disposeIntermediateTensorInfo(p),n.disposeIntermediateTensorInfo(m),h}let hm={kernelName:xn,backendName:`cpu`,kernelFunc:mm};
/**
* @license
* Copyright 2021 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function gm(e){let{inputs:t,backend:n}=e,{indices:r,values:i,denseShape:a,defaultValue:o}=t;if(a.shape.length!==1)throw Error(`Dense shape must be a vector, saw:
        ${a.shape}`);if(r.shape.length!==2)throw Error(`Indices must be a matrix, saw:
        ${r.shape}`);if(i.shape.length!==1)throw Error(`Values must be a vector, saw:
        ${i.shape}`);if(o.shape.length!==0)throw Error(`Default value must be a scalar, saw:
        ${o.shape}`);let s=n.data.get(r.dataId).values,c=n.data.get(i.dataId).values,l=n.data.get(a.dataId).values,u=n.data.get(o.dataId).values[0],[d,f,p,m,h]=Hl(s,r.shape,r.dtype,c,i.dtype,l,u);return[n.makeTensorInfo(f,r.dtype,d),n.makeTensorInfo([f[0]],i.dtype,p),n.makeTensorInfo([m.length],`bool`,new Uint8Array(m.map(e=>Number(e)))),n.makeTensorInfo([h.length],r.dtype,new Int32Array(h))]}let _m={kernelName:wn,backendName:`cpu`,kernelFunc:gm};
/**
* @license
* Copyright 2021 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function vm(e){let{inputs:t,backend:n}=e,{inputIndices:r,inputShape:i,newShape:a}=t;if(r.shape.length!==2)throw Error(`Input indices should be a matrix but received shape
        ${r.shape}`);if(i.shape.length!==1)throw Error(`Input shape should be a vector but received shape
        ${i.shape}`);if(a.shape.length!==1)throw Error(`Target shape should be a vector but received shape ${a.shape}`);let o=Array.from(n.data.get(i.dataId).values),s=n.data.get(r.dataId).values,c=Array.from(n.data.get(a.dataId).values),[l,u,d]=Ul(s,r.shape,r.dtype,o,c);return[n.makeTensorInfo(u,r.dtype,l),n.makeTensorInfo([d.length],a.dtype,new Int32Array(d))]}let ym={kernelName:Tn,backendName:`cpu`,kernelFunc:vm};
/**
* @license
* Copyright 2021 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function bm(e){let{inputs:t,backend:n}=e,{data:r,indices:i,segmentIds:a}=t;if(r.shape.length<1)throw Error(`Data should be at least 1 dimensional but received scalar`);if(i.shape.length!==1)throw Error(`Indices should be a vector but received shape
          ${i.shape}`);if(a.shape.length!==1)throw Error(`Segment ids should be a vector but received shape
          ${a.shape}`);if(i.shape[0]!==a.shape[0])throw Error(`segmentIds and indices should have same size.`);let o=n.data.get(r.dataId).values,s=n.data.get(i.dataId).values,c=n.data.get(a.dataId).values,[l,u]=Wl(o,r.shape,r.dtype,s,c,!0);return n.makeTensorInfo(u,r.dtype,l)}let xm={kernelName:En,backendName:`cpu`,kernelFunc:bm};
/**
* @license
* Copyright 2021 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Sm(e){let{inputs:t,backend:n}=e,{data:r,indices:i,segmentIds:a}=t;if(r.shape.length<1)throw Error(`Data should be at least 1 dimensional but received scalar`);if(i.shape.length!==1)throw Error(`Indices should be a vector but received shape
         ${i.shape}`);if(a.shape.length!==1)throw Error(`Segment ids should be a vector but received shape
         ${a.shape}`);if(i.shape[0]!==a.shape[0])throw Error(`segmentIds and indices should have same size.`);let o=n.data.get(r.dataId).values,s=n.data.get(i.dataId).values,c=n.data.get(a.dataId).values,[l,u]=Wl(o,r.shape,r.dtype,s,c);return n.makeTensorInfo(u,r.dtype,l)}let Cm={kernelName:Dn,backendName:`cpu`,kernelFunc:Sm};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function wm(e){let{inputs:t,backend:n,attrs:r}=e,{sparseIndices:i,sparseValues:a,defaultValue:o}=t,{outputShape:s}=r,{sliceRank:c,numUpdates:l,sliceSize:u,strides:d,outputSize:f}=Wo(a,i,s),p=n.bufferSync(i),m;switch(a.dtype){case`bool`:m=Il(p,n.bufferSync(a),s,f,u,l,c,d,!!n.data.get(o.dataId).values[0],!1);break;case`float32`:{let e=n.bufferSync(a),t=n.data.get(o.dataId).values[0];m=Il(p,e,s,f,u,l,c,d,t,!1);break}case`int32`:{let e=n.bufferSync(a),t=n.data.get(o.dataId).values[0];m=Il(p,e,s,f,u,l,c,d,t,!1);break}case`string`:m=Il(p,n.bufferSync(a),s,f,u,l,c,d,Dr(n.data.get(o.dataId).values[0]),!1);break;default:throw Error(`Unsupported type ${a.dtype}`)}return n.makeTensorInfo(s,m.dtype,m.values)}let Tm={kernelName:On,backendName:`cpu`,kernelFunc:wm};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Em(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{numOrSizeSplits:a,axis:o}=r,s=S(o,i.shape)[0],c=$s(i,a,s),l=Array(i.shape.length).fill(0),u=i.shape.slice();return c.map(e=>{let t=[...u];t[s]=e;let r=Bl({inputs:{x:i},backend:n,attrs:{begin:l,size:t}});return l[s]+=e,r})}let Dm={kernelName:Sn,backendName:`cpu`,kernelFunc:Em},Om={kernelName:An,backendName:`cpu`,kernelFunc:({inputs:e,backend:t})=>{
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let{x:n}=e,r=t;q(n,`square`);let i=r.data.get(n.dataId).values,a=new Float32Array(i.length);for(let e=0;e<i.length;++e){let t=i[e];a[e]=t*t}return{dataId:r.write(a,n.shape,n.dtype),shape:n.shape,dtype:n.dtype}}},km={kernelName:Wn,backendName:`cpu`,kernelFunc:J(Wn,(e,t)=>isNaN(e)?NaN:e>0?1:t.alpha)}
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the License);
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an AS IS BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Am(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{begin:a,end:o,strides:s,beginMask:c,endMask:l,ellipsisMask:u,newAxisMask:d,shrinkAxisMask:f}=r;q(i,`stridedSlice`);let{finalShapeSparse:p,finalShape:h,isIdentity:g,sliceDim0:_,isSimpleSlice:v,begin:y,end:b,strides:x}=Ss(i.shape,a,o,s,c,l,u,d,f),S;if(g)S=yu({inputs:{x:i},backend:n,attrs:{shape:h}});else if(_||v){m(i.shape.length>=1,()=>`Input must have rank at least 1, got: ${i.shape.length}`);let e=vs(y,b,x),t=Bl({inputs:{x:i},backend:n,attrs:{begin:y,size:e}});S=yu({inputs:{x:t},backend:n,attrs:{shape:h}}),n.disposeIntermediateTensorInfo(t)}else{let e=Jl(p,n.bufferSync(i),x,y);S=n.makeTensorInfo(h,e.dtype,e.values)}return S}let jm={kernelName:Mn,backendName:`cpu`,kernelFunc:Am};
/**
* @license
* Copyright 2021 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Mm(e){let{inputs:t,backend:n,attrs:r}=e,{separator:i,nGramWidths:a,leftPad:o,rightPad:s,padWidth:c,preserveShortSequences:l}=r,{data:u,dataSplits:d}=t,f=n.data.get(u.dataId).values,p=n.data.get(d.dataId).values,[m,h]=Xl(f,p,i,a,o,s,c,l);return[n.makeTensorInfo([m.length],`string`,m),n.makeTensorInfo(d.shape,`int32`,h)]}let Nm={kernelName:Nn,backendName:`cpu`,kernelFunc:Mm};
/**
* @license
* Copyright 2021 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Pm(e){let{inputs:t,backend:n,attrs:r}=e,{skipEmpty:i}=r,{input:a,delimiter:o}=t;if(a.dtype!==`string`)throw Error(`Input must be of datatype string`);if(a.shape.length!==1)throw Error(`Input must be a vector, got shape: ${a.shape}`);if(o.shape.length!==0)throw Error(`Delimiter must be a scalar, got shape: ${o.shape}`);let s=n.data.get(a.dataId).values,c=n.data.get(o.dataId).values[0],[l,u,d]=Ql(s,c,i),f=u.length;return[n.makeTensorInfo([f,2],`int32`,l),n.makeTensorInfo([f],`string`,u),n.makeTensorInfo([2],`int32`,new Int32Array(d))]}let Fm={kernelName:Pn,backendName:`cpu`,kernelFunc:Pm};
/**
* @license
* Copyright 2021 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Im(e){let{inputs:t,backend:n,attrs:r}=e,{numBuckets:i}=r,{input:a}=t;if(a.dtype!==`string`)throw Error(`Input must be of datatype string`);if(i<=0)throw Error(`Number of buckets must be at least 1`);let o=n.data.get(a.dataId).values,s=$l(o,i);return n.makeTensorInfo(a.shape,`int32`,s)}let Lm={kernelName:Fn,backendName:`cpu`,kernelFunc:Im},Rm={kernelName:`Tan`,backendName:`cpu`,kernelFunc:J(`Tan`,e=>Math.tan(e))},zm={kernelName:In,backendName:`cpu`,kernelFunc:J(In,e=>Math.tanh(e))}
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the License);
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an AS IS BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the License);
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an AS IS BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;
/**
* @license
* Copyright 2022 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Bm(e){let{inputs:t,backend:n}=e,{tensor:r,indices:i,updates:a}=t,{sliceRank:o,numUpdates:s,sliceSize:c,strides:l,outputSize:u}=Wo(a,i,r.shape),d=n.bufferSync(i),f=n.bufferSync(a),p=n.bufferSync(r),m=Il(d,f,r.shape,u,c,s,o,l,p,!1);return n.makeTensorInfo(r.shape,m.dtype,m.values)}let Vm={kernelName:dn,backendName:`cpu`,kernelFunc:Bm};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Hm(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{reps:a}=r;q(i,`tile`);let o=nu(n.bufferSync(i),a);return n.makeTensorInfo(o.shape,o.dtype,o.values)}let Um={kernelName:Ln,backendName:`cpu`,kernelFunc:Hm};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Wm(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{k:a,sorted:o}=r;q(i,`topk`);let s=n.data.get(i.dataId).values,[c,l]=au(s,i.shape,i.dtype,a,o);return[n.makeTensorInfo(c.shape,c.dtype,c.values),n.makeTensorInfo(l.shape,l.dtype,l.values)]}let Gm={kernelName:Rn,backendName:`cpu`,kernelFunc:Wm};
/**
* @license
* Copyright 2021 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Km(e){let{inputs:t,attrs:n,backend:r}=e,{image:i,transforms:a}=t,{interpolation:o,fillMode:s,fillValue:c,outputShape:l}=n,[u,d,f,p]=i.shape,[m,h]=l??[d,f],g=[u,m,h,p],v=P(i.shape),y=v[0],b=v[1],x=v[2],S=P(g),w=S[0],T=S[1],E=S[2],D=C(i.dtype,_(g));D.fill(c);let O=r.data.get(i.dataId).values,k=r.data.get(a.dataId).values;for(let e=0;e<u;++e){let t=a.shape[0]===1?k:k.subarray(e*8,e*8+8);for(let n=0;n<m;++n)for(let r=0;r<h;++r)for(let i=0;i<p;++i){let a,l=t[6]*r+t[7]*n+1;if(l===0)continue;let u=(t[0]*r+t[1]*n+t[2])/l,p=(t[3]*r+t[4]*n+t[5])/l,m=Jm(u,f,s),h=Jm(p,d,s);switch(o){case`nearest`:a=eh(O,d,f,y,b,x,e,h,m,i,c);break;case`bilinear`:a=th(O,d,f,y,b,x,e,h,m,i,c);break;default:throw Error(`Error in Transform: Expect 'nearest' or 'bilinear', but got ${o}`)}let g=e*w+n*T+r*E+i;D[g]=a}return r.makeTensorInfo(g,i.dtype,D)}return{dataId:r.write(D,g,i.dtype),shape:i.shape,dtype:i.dtype}}let qm={kernelName:zn,backendName:`cpu`,kernelFunc:Km};function Jm(e,t,n){switch(n){case`reflect`:return Ym(e,t);case`wrap`:return Xm(e,t);case`nearest`:return Qm(e,t);default:return Zm(e,t)}}function Ym(e,t){let n=e;if(n<0){if(t<=1)n=0;else{let e=2*t;n<e&&(n=e*Math.trunc(-n/e)+n),n=n<-t?n+e:-n-1}}else if(n>t-1){if(t<=1)n=0;else{let e=2*t;n-=e*Math.trunc(n/e),n>=t&&(n=e-n-1)}}return f(0,n,t-1)}function Xm(e,t){let n=e;if(n<0){if(t<=1)n=0;else{let e=t-1;n+=t*(Math.trunc(-n/e)+1)}}else if(n>t-1){if(t<=1)n=0;else{let e=t-1;n-=t*Math.trunc(n/e)}}return f(0,n,t-1)}function Zm(e,t){return e}function Qm(e,t){return f(0,e,t-1)}function $m(e,t,n,r,i,a,o,s,c,l,u){let d=o*r+s*i+c*a+l;return 0<=s&&s<t&&0<=c&&c<n?e[d]:u}function eh(e,t,n,r,i,a,o,s,c,l,u){return $m(e,t,n,r,i,a,o,Math.round(s),Math.round(c),l,u)}function th(e,t,n,r,i,a,o,s,c,l,u){let d=Math.floor(s),f=Math.floor(c),p=d+1,m=f+1,h=(m-c)*$m(e,t,n,r,i,a,o,d,f,l,u)+(c-f)*$m(e,t,n,r,i,a,o,d,m,l,u),g=(m-c)*$m(e,t,n,r,i,a,o,p,f,l,u)+(c-f)*$m(e,t,n,r,i,a,o,p,m,l,u);return(p-s)*h+(s-d)*g}
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the License);
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an AS IS BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function nh(e){let{inputs:t,attrs:n,backend:r}=e,{axis:i}=n,{x:a}=t;q(a,`unique`);let o=r.data.get(a.dataId).values,{outputValues:s,outputShape:c,indices:l}=ou(o,i,a.shape,a.dtype);return[r.makeTensorInfo(c,a.dtype,s),r.makeTensorInfo([l.length],`int32`,l)]}let rh={kernelName:Vn,backendName:`cpu`,kernelFunc:nh};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function ih(e){let{inputs:t,backend:n,attrs:r}=e,{value:i}=t,{axis:a}=r;a<0&&(a+=i.shape.length);let o=i.shape.length,s=i.shape[a],c=Array(o-1),l=0;for(let e=0;e<o;e++)e!==a&&(c[l++]=i.shape[e]);let u=Array(o).fill(0),d=i.shape.slice();d[a]=1;let f=Array(s);for(let e=0;e<f.length;e++){u[a]=e;let t=Bl({inputs:{x:i},backend:n,attrs:{begin:u,size:d}});f[e]=yu({inputs:{x:t},backend:n,attrs:{shape:c}}),n.disposeIntermediateTensorInfo(t)}return f}let ah={kernelName:Hn,backendName:`cpu`,kernelFunc:ih};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function oh(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,segmentIds:a}=t,{numSegments:o}=r;q(i,`unsortedSegmentSum`);let s=i.shape.length,c=a.shape.length,l=[],u=[],d=s-c,f=a;for(let e=0;e<d;++e){let t=af({inputs:{input:f},backend:n,attrs:{dim:e+1}});f=t,u.push(t)}for(let e=0;e<o;++e){let t=xr(e,`int32`),r=n.makeTensorInfo([],`int32`,t),a=Vc({inputs:{a:r,b:f},backend:n}),o=Oc({inputs:{x:a},backend:n,attrs:{dtype:`float32`}}),s=ol({inputs:{a:o,b:i},backend:n}),c=Zd({inputs:{x:s},backend:n,attrs:{axis:0,keepDims:!1}});l.push(c),u.push(r),u.push(a),u.push(o),u.push(s),u.push(c)}let p=Dp({inputs:l,backend:n,attrs:{axis:0}});return u.forEach(e=>n.disposeIntermediateTensorInfo(e)),p}
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let sh=[wu,vc,Tu,Eu,Nc,Ou,Au,Mu,Pu,Iu,Lu,Ru,zu,Bu,Vu,qu,Yu,Zu,$u,Su,td,rd,ad,Ic,sd,kc,zc,cd,xc,ld,pd,hd,_d,yd,xd,Cd,Td,Ed,Dd,kd,jd,Nd,Fd,Ld,zd,Vd,Ud,Gd,Kd,qd,Jd,Xd,ef,cu,nf,Hc,rf,Wc,of,Gc,hf,_f,yf,Kc,qc,xf,Cf,Tf,Df,Xc,Zc,wc,kf,dd,Af,jf,Mf,uu,Qc,$c,Pf,tl,Ff,If,Lf,Rf,Bf,Hf,Wf,rl,Kf,Jf,Xf,Qf,ep,np,ip,il,op,sp,dp,sl,dl,mp,_p,bp,fl,Sp,Ep,Op,Ap,jp,pu,vl,Np,Fp,Lp,zp,Ec,cf,Bp,hu,_u,bu,Hp,Wp,Kp,Jp,Xp,Zp,Qp,Fl,em,am,sm,cm,Rl,lm,um,dm,Vl,lp,pm,hm,_m,ym,xm,Cm,Tm,Dm,Gl,Om,Kl,ql,km,jm,Nm,Fm,Lm,tu,Qd,Rm,zm,Vm,Um,Gm,qm,hl,rh,ah,{kernelName:`UnsortedSegmentSum`,backendName:`cpu`,kernelFunc:oh},wp];for(let e of sh)tr(e)
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/*! pako 2.1.0 https://github.com/nodeca/pako @license (MIT AND Zlib) */
;function ch(e){let t=e.length;for(;--t>=0;)e[t]=0}let lh=new Uint8Array([0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0]),uh=new Uint8Array([0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13]),dh=new Uint8Array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,3,7]),fh=new Uint8Array([16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15]),ph=Array(576);ch(ph);let mh=Array(60);ch(mh);let hh=Array(512);ch(hh);let gh=Array(256);ch(gh);let _h=Array(29);ch(_h);let vh=Array(30);ch(vh);function yh(e,t,n,r,i){this.static_tree=e,this.extra_bits=t,this.extra_base=n,this.elems=r,this.max_length=i,this.has_stree=e&&e.length}let bh,xh,Sh;function Ch(e,t){this.dyn_tree=e,this.max_code=0,this.stat_desc=t}let wh=e=>e<256?hh[e]:hh[256+(e>>>7)],Th=(e,t)=>{e.pending_buf[e.pending++]=t&255,e.pending_buf[e.pending++]=t>>>8&255},Eh=(e,t,n)=>{e.bi_valid>16-n?(e.bi_buf|=t<<e.bi_valid&65535,Th(e,e.bi_buf),e.bi_buf=t>>16-e.bi_valid,e.bi_valid+=n-16):(e.bi_buf|=t<<e.bi_valid&65535,e.bi_valid+=n)},Dh=(e,t,n)=>{Eh(e,n[t*2],n[t*2+1])},Oh=(e,t)=>{let n=0;do n|=e&1,e>>>=1,n<<=1;while(--t>0);return n>>>1},kh=e=>{e.bi_valid===16?(Th(e,e.bi_buf),e.bi_buf=0,e.bi_valid=0):e.bi_valid>=8&&(e.pending_buf[e.pending++]=e.bi_buf&255,e.bi_buf>>=8,e.bi_valid-=8)},Ah=(e,t)=>{let n=t.dyn_tree,r=t.max_code,i=t.stat_desc.static_tree,a=t.stat_desc.has_stree,o=t.stat_desc.extra_bits,s=t.stat_desc.extra_base,c=t.stat_desc.max_length,l,u,d,f,p,m,h=0;for(f=0;f<=15;f++)e.bl_count[f]=0;for(n[e.heap[e.heap_max]*2+1]=0,l=e.heap_max+1;l<573;l++)u=e.heap[l],f=n[n[u*2+1]*2+1]+1,f>c&&(f=c,h++),n[u*2+1]=f,!(u>r)&&(e.bl_count[f]++,p=0,u>=s&&(p=o[u-s]),m=n[u*2],e.opt_len+=m*(f+p),a&&(e.static_len+=m*(i[u*2+1]+p)));if(h!==0){do{for(f=c-1;e.bl_count[f]===0;)f--;e.bl_count[f]--,e.bl_count[f+1]+=2,e.bl_count[c]--,h-=2}while(h>0);for(f=c;f!==0;f--)for(u=e.bl_count[f];u!==0;)d=e.heap[--l],!(d>r)&&(n[d*2+1]!==f&&(e.opt_len+=(f-n[d*2+1])*n[d*2],n[d*2+1]=f),u--)}},jh=(e,t,n)=>{let r=Array(16),i=0,a,o;for(a=1;a<=15;a++)i=i+n[a-1]<<1,r[a]=i;for(o=0;o<=t;o++){let t=e[o*2+1];t!==0&&(e[o*2]=Oh(r[t]++,t))}},Mh=()=>{let e,t,n,r,i,a=Array(16);for(n=0,r=0;r<28;r++)for(_h[r]=n,e=0;e<1<<lh[r];e++)gh[n++]=r;for(gh[n-1]=r,i=0,r=0;r<16;r++)for(vh[r]=i,e=0;e<1<<uh[r];e++)hh[i++]=r;for(i>>=7;r<30;r++)for(vh[r]=i<<7,e=0;e<1<<uh[r]-7;e++)hh[256+i++]=r;for(t=0;t<=15;t++)a[t]=0;for(e=0;e<=143;)ph[e*2+1]=8,e++,a[8]++;for(;e<=255;)ph[e*2+1]=9,e++,a[9]++;for(;e<=279;)ph[e*2+1]=7,e++,a[7]++;for(;e<=287;)ph[e*2+1]=8,e++,a[8]++;for(jh(ph,287,a),e=0;e<30;e++)mh[e*2+1]=5,mh[e*2]=Oh(e,5);bh=new yh(ph,lh,257,286,15),xh=new yh(mh,uh,0,30,15),Sh=new yh([],dh,0,19,7)},Nh=e=>{let t;for(t=0;t<286;t++)e.dyn_ltree[t*2]=0;for(t=0;t<30;t++)e.dyn_dtree[t*2]=0;for(t=0;t<19;t++)e.bl_tree[t*2]=0;e.dyn_ltree[512]=1,e.opt_len=e.static_len=0,e.sym_next=e.matches=0},Ph=e=>{e.bi_valid>8?Th(e,e.bi_buf):e.bi_valid>0&&(e.pending_buf[e.pending++]=e.bi_buf),e.bi_buf=0,e.bi_valid=0},Fh=(e,t,n,r)=>{let i=t*2,a=n*2;return e[i]<e[a]||e[i]===e[a]&&r[t]<=r[n]},Ih=(e,t,n)=>{let r=e.heap[n],i=n<<1;for(;i<=e.heap_len&&(i<e.heap_len&&Fh(t,e.heap[i+1],e.heap[i],e.depth)&&i++,!Fh(t,r,e.heap[i],e.depth));)e.heap[n]=e.heap[i],n=i,i<<=1;e.heap[n]=r},Lh=(e,t,n)=>{let r,i,a=0,o,s;if(e.sym_next!==0)do r=e.pending_buf[e.sym_buf+a++]&255,r+=(e.pending_buf[e.sym_buf+a++]&255)<<8,i=e.pending_buf[e.sym_buf+a++],r===0?Dh(e,i,t):(o=gh[i],Dh(e,o+256+1,t),s=lh[o],s!==0&&(i-=_h[o],Eh(e,i,s)),r--,o=wh(r),Dh(e,o,n),s=uh[o],s!==0&&(r-=vh[o],Eh(e,r,s)));while(a<e.sym_next);Dh(e,256,t)},Rh=(e,t)=>{let n=t.dyn_tree,r=t.stat_desc.static_tree,i=t.stat_desc.has_stree,a=t.stat_desc.elems,o,s,c=-1,l;for(e.heap_len=0,e.heap_max=573,o=0;o<a;o++)n[o*2]===0?n[o*2+1]=0:(e.heap[++e.heap_len]=c=o,e.depth[o]=0);for(;e.heap_len<2;)l=e.heap[++e.heap_len]=c<2?++c:0,n[l*2]=1,e.depth[l]=0,e.opt_len--,i&&(e.static_len-=r[l*2+1]);for(t.max_code=c,o=e.heap_len>>1;o>=1;o--)Ih(e,n,o);l=a;do o=e.heap[1],e.heap[1]=e.heap[e.heap_len--],Ih(e,n,1),s=e.heap[1],e.heap[--e.heap_max]=o,e.heap[--e.heap_max]=s,n[l*2]=n[o*2]+n[s*2],e.depth[l]=(e.depth[o]>=e.depth[s]?e.depth[o]:e.depth[s])+1,n[o*2+1]=n[s*2+1]=l,e.heap[1]=l++,Ih(e,n,1);while(e.heap_len>=2);e.heap[--e.heap_max]=e.heap[1],Ah(e,t),jh(n,c,e.bl_count)},zh=(e,t,n)=>{let r,i=-1,a,o=t[1],s=0,c=7,l=4;for(o===0&&(c=138,l=3),t[(n+1)*2+1]=65535,r=0;r<=n;r++)a=o,o=t[(r+1)*2+1],!(++s<c&&a===o)&&(s<l?e.bl_tree[a*2]+=s:a===0?s<=10?e.bl_tree[34]++:e.bl_tree[36]++:(a!==i&&e.bl_tree[a*2]++,e.bl_tree[32]++),s=0,i=a,o===0?(c=138,l=3):a===o?(c=6,l=3):(c=7,l=4))},Bh=(e,t,n)=>{let r,i=-1,a,o=t[1],s=0,c=7,l=4;for(o===0&&(c=138,l=3),r=0;r<=n;r++)if(a=o,o=t[(r+1)*2+1],!(++s<c&&a===o)){if(s<l)do Dh(e,a,e.bl_tree);while(--s!==0);else a===0?s<=10?(Dh(e,17,e.bl_tree),Eh(e,s-3,3)):(Dh(e,18,e.bl_tree),Eh(e,s-11,7)):(a!==i&&(Dh(e,a,e.bl_tree),s--),Dh(e,16,e.bl_tree),Eh(e,s-3,2));s=0,i=a,o===0?(c=138,l=3):a===o?(c=6,l=3):(c=7,l=4)}},Vh=e=>{let t;for(zh(e,e.dyn_ltree,e.l_desc.max_code),zh(e,e.dyn_dtree,e.d_desc.max_code),Rh(e,e.bl_desc),t=18;t>=3&&e.bl_tree[fh[t]*2+1]===0;t--);return e.opt_len+=3*(t+1)+5+5+4,t},Hh=(e,t,n,r)=>{let i;for(Eh(e,t-257,5),Eh(e,n-1,5),Eh(e,r-4,4),i=0;i<r;i++)Eh(e,e.bl_tree[fh[i]*2+1],3);Bh(e,e.dyn_ltree,t-1),Bh(e,e.dyn_dtree,n-1)},Uh=e=>{let t=4093624447,n;for(n=0;n<=31;n++,t>>>=1)if(t&1&&e.dyn_ltree[n*2]!==0)return 0;if(e.dyn_ltree[18]!==0||e.dyn_ltree[20]!==0||e.dyn_ltree[26]!==0)return 1;for(n=32;n<256;n++)if(e.dyn_ltree[n*2]!==0)return 1;return 0},Wh=!1,Gh=e=>{Wh||=(Mh(),!0),e.l_desc=new Ch(e.dyn_ltree,bh),e.d_desc=new Ch(e.dyn_dtree,xh),e.bl_desc=new Ch(e.bl_tree,Sh),e.bi_buf=0,e.bi_valid=0,Nh(e)},Kh=(e,t,n,r)=>{Eh(e,0+ +!!r,3),Ph(e),Th(e,n),Th(e,~n),n&&e.pending_buf.set(e.window.subarray(t,t+n),e.pending),e.pending+=n};var qh={_tr_init:Gh,_tr_stored_block:Kh,_tr_flush_block:(e,t,n,r)=>{let i,a,o=0;e.level>0?(e.strm.data_type===2&&(e.strm.data_type=Uh(e)),Rh(e,e.l_desc),Rh(e,e.d_desc),o=Vh(e),i=e.opt_len+3+7>>>3,a=e.static_len+3+7>>>3,a<=i&&(i=a)):i=a=n+5,n+4<=i&&t!==-1?Kh(e,t,n,r):e.strategy===4||a===i?(Eh(e,2+ +!!r,3),Lh(e,ph,mh)):(Eh(e,4+ +!!r,3),Hh(e,e.l_desc.max_code+1,e.d_desc.max_code+1,o+1),Lh(e,e.dyn_ltree,e.dyn_dtree)),Nh(e),r&&Ph(e)},_tr_tally:(e,t,n)=>(e.pending_buf[e.sym_buf+e.sym_next++]=t,e.pending_buf[e.sym_buf+e.sym_next++]=t>>8,e.pending_buf[e.sym_buf+e.sym_next++]=n,t===0?e.dyn_ltree[n*2]++:(e.matches++,t--,e.dyn_ltree[(gh[n]+256+1)*2]++,e.dyn_dtree[wh(t)*2]++),e.sym_next===e.sym_end),_tr_align:e=>{Eh(e,2,3),Dh(e,256,ph),kh(e)}},Jh=(e,t,n,r)=>{let i=e&65535|0,a=e>>>16&65535|0,o=0;for(;n!==0;){o=n>2e3?2e3:n,n-=o;do i=i+t[r++]|0,a=a+i|0;while(--o);i%=65521,a%=65521}return i|a<<16|0};let Yh=new Uint32Array((()=>{let e,t=[];for(var n=0;n<256;n++){e=n;for(var r=0;r<8;r++)e=e&1?3988292384^e>>>1:e>>>1;t[n]=e}return t})());var Xh=(e,t,n,r)=>{let i=Yh,a=r+n;e^=-1;for(let n=r;n<a;n++)e=e>>>8^i[(e^t[n])&255];return e^-1},Zh={2:`need dictionary`,1:`stream end`,0:``,"-1":`file error`,"-2":`stream error`,"-3":`data error`,"-4":`insufficient memory`,"-5":`buffer error`,"-6":`incompatible version`},Qh={Z_NO_FLUSH:0,Z_PARTIAL_FLUSH:1,Z_SYNC_FLUSH:2,Z_FULL_FLUSH:3,Z_FINISH:4,Z_BLOCK:5,Z_TREES:6,Z_OK:0,Z_STREAM_END:1,Z_NEED_DICT:2,Z_ERRNO:-1,Z_STREAM_ERROR:-2,Z_DATA_ERROR:-3,Z_MEM_ERROR:-4,Z_BUF_ERROR:-5,Z_NO_COMPRESSION:0,Z_BEST_SPEED:1,Z_BEST_COMPRESSION:9,Z_DEFAULT_COMPRESSION:-1,Z_FILTERED:1,Z_HUFFMAN_ONLY:2,Z_RLE:3,Z_FIXED:4,Z_DEFAULT_STRATEGY:0,Z_BINARY:0,Z_TEXT:1,Z_UNKNOWN:2,Z_DEFLATED:8};let{_tr_init:$h,_tr_stored_block:eg,_tr_flush_block:tg,_tr_tally:ng,_tr_align:rg}=qh,{Z_NO_FLUSH:ig,Z_PARTIAL_FLUSH:ag,Z_FULL_FLUSH:og,Z_FINISH:sg,Z_BLOCK:cg,Z_OK:lg,Z_STREAM_END:ug,Z_STREAM_ERROR:dg,Z_DATA_ERROR:fg,Z_BUF_ERROR:pg,Z_DEFAULT_COMPRESSION:mg,Z_FILTERED:hg,Z_HUFFMAN_ONLY:gg,Z_RLE:_g,Z_FIXED:vg,Z_DEFAULT_STRATEGY:yg,Z_UNKNOWN:bg,Z_DEFLATED:xg}=Qh,Sg=(e,t)=>(e.msg=Zh[t],t),Cg=e=>e*2-(e>4?9:0),wg=e=>{let t=e.length;for(;--t>=0;)e[t]=0},Tg=e=>{let t,n,r,i=e.w_size;t=e.hash_size,r=t;do n=e.head[--r],e.head[r]=n>=i?n-i:0;while(--t);t=i,r=t;do n=e.prev[--r],e.prev[r]=n>=i?n-i:0;while(--t)},Eg=(e,t,n)=>(t<<e.hash_shift^n)&e.hash_mask,Dg=e=>{let t=e.state,n=t.pending;n>e.avail_out&&(n=e.avail_out),n!==0&&(e.output.set(t.pending_buf.subarray(t.pending_out,t.pending_out+n),e.next_out),e.next_out+=n,t.pending_out+=n,e.total_out+=n,e.avail_out-=n,t.pending-=n,t.pending===0&&(t.pending_out=0))},Og=(e,t)=>{tg(e,e.block_start>=0?e.block_start:-1,e.strstart-e.block_start,t),e.block_start=e.strstart,Dg(e.strm)},Y=(e,t)=>{e.pending_buf[e.pending++]=t},kg=(e,t)=>{e.pending_buf[e.pending++]=t>>>8&255,e.pending_buf[e.pending++]=t&255},Ag=(e,t,n,r)=>{let i=e.avail_in;return i>r&&(i=r),i===0?0:(e.avail_in-=i,t.set(e.input.subarray(e.next_in,e.next_in+i),n),e.state.wrap===1?e.adler=Jh(e.adler,t,i,n):e.state.wrap===2&&(e.adler=Xh(e.adler,t,i,n)),e.next_in+=i,e.total_in+=i,i)},jg=(e,t)=>{let n=e.max_chain_length,r=e.strstart,i,a,o=e.prev_length,s=e.nice_match,c=e.strstart>e.w_size-262?e.strstart-(e.w_size-262):0,l=e.window,u=e.w_mask,d=e.prev,f=e.strstart+258,p=l[r+o-1],m=l[r+o];e.prev_length>=e.good_match&&(n>>=2),s>e.lookahead&&(s=e.lookahead);do if(i=t,l[i+o]===m&&l[i+o-1]===p&&l[i]===l[r]&&l[++i]===l[r+1]){r+=2,i++;do;while(l[++r]===l[++i]&&l[++r]===l[++i]&&l[++r]===l[++i]&&l[++r]===l[++i]&&l[++r]===l[++i]&&l[++r]===l[++i]&&l[++r]===l[++i]&&l[++r]===l[++i]&&r<f);if(a=258-(f-r),r=f-258,a>o){if(e.match_start=t,o=a,a>=s)break;p=l[r+o-1],m=l[r+o]}}while((t=d[t&u])>c&&--n!==0);return o<=e.lookahead?o:e.lookahead},Mg=e=>{let t=e.w_size,n,r,i;do{if(r=e.window_size-e.lookahead-e.strstart,e.strstart>=t+(t-262)&&(e.window.set(e.window.subarray(t,t+t-r),0),e.match_start-=t,e.strstart-=t,e.block_start-=t,e.insert>e.strstart&&(e.insert=e.strstart),Tg(e),r+=t),e.strm.avail_in===0)break;if(n=Ag(e.strm,e.window,e.strstart+e.lookahead,r),e.lookahead+=n,e.lookahead+e.insert>=3)for(i=e.strstart-e.insert,e.ins_h=e.window[i],e.ins_h=Eg(e,e.ins_h,e.window[i+1]);e.insert&&(e.ins_h=Eg(e,e.ins_h,e.window[i+3-1]),e.prev[i&e.w_mask]=e.head[e.ins_h],e.head[e.ins_h]=i,i++,e.insert--,!(e.lookahead+e.insert<3)););}while(e.lookahead<262&&e.strm.avail_in!==0)},Ng=(e,t)=>{let n=e.pending_buf_size-5>e.w_size?e.w_size:e.pending_buf_size-5,r,i,a,o=0,s=e.strm.avail_in;do{if(r=65535,a=e.bi_valid+42>>3,e.strm.avail_out<a||(a=e.strm.avail_out-a,i=e.strstart-e.block_start,r>i+e.strm.avail_in&&(r=i+e.strm.avail_in),r>a&&(r=a),r<n&&(r===0&&t!==sg||t===ig||r!==i+e.strm.avail_in)))break;o=+(t===sg&&r===i+e.strm.avail_in),eg(e,0,0,o),e.pending_buf[e.pending-4]=r,e.pending_buf[e.pending-3]=r>>8,e.pending_buf[e.pending-2]=~r,e.pending_buf[e.pending-1]=~r>>8,Dg(e.strm),i&&(i>r&&(i=r),e.strm.output.set(e.window.subarray(e.block_start,e.block_start+i),e.strm.next_out),e.strm.next_out+=i,e.strm.avail_out-=i,e.strm.total_out+=i,e.block_start+=i,r-=i),r&&(Ag(e.strm,e.strm.output,e.strm.next_out,r),e.strm.next_out+=r,e.strm.avail_out-=r,e.strm.total_out+=r)}while(o===0);return s-=e.strm.avail_in,s&&(s>=e.w_size?(e.matches=2,e.window.set(e.strm.input.subarray(e.strm.next_in-e.w_size,e.strm.next_in),0),e.strstart=e.w_size,e.insert=e.strstart):(e.window_size-e.strstart<=s&&(e.strstart-=e.w_size,e.window.set(e.window.subarray(e.w_size,e.w_size+e.strstart),0),e.matches<2&&e.matches++,e.insert>e.strstart&&(e.insert=e.strstart)),e.window.set(e.strm.input.subarray(e.strm.next_in-s,e.strm.next_in),e.strstart),e.strstart+=s,e.insert+=s>e.w_size-e.insert?e.w_size-e.insert:s),e.block_start=e.strstart),e.high_water<e.strstart&&(e.high_water=e.strstart),o?4:t!==ig&&t!==sg&&e.strm.avail_in===0&&e.strstart===e.block_start?2:(a=e.window_size-e.strstart,e.strm.avail_in>a&&e.block_start>=e.w_size&&(e.block_start-=e.w_size,e.strstart-=e.w_size,e.window.set(e.window.subarray(e.w_size,e.w_size+e.strstart),0),e.matches<2&&e.matches++,a+=e.w_size,e.insert>e.strstart&&(e.insert=e.strstart)),a>e.strm.avail_in&&(a=e.strm.avail_in),a&&(Ag(e.strm,e.window,e.strstart,a),e.strstart+=a,e.insert+=a>e.w_size-e.insert?e.w_size-e.insert:a),e.high_water<e.strstart&&(e.high_water=e.strstart),a=e.bi_valid+42>>3,a=e.pending_buf_size-a>65535?65535:e.pending_buf_size-a,n=a>e.w_size?e.w_size:a,i=e.strstart-e.block_start,(i>=n||(i||t===sg)&&t!==ig&&e.strm.avail_in===0&&i<=a)&&(r=i>a?a:i,o=+(t===sg&&e.strm.avail_in===0&&r===i),eg(e,e.block_start,r,o),e.block_start+=r,Dg(e.strm)),o?3:1)},Pg=(e,t)=>{let n,r;for(;;){if(e.lookahead<262){if(Mg(e),e.lookahead<262&&t===ig)return 1;if(e.lookahead===0)break}if(n=0,e.lookahead>=3&&(e.ins_h=Eg(e,e.ins_h,e.window[e.strstart+3-1]),n=e.prev[e.strstart&e.w_mask]=e.head[e.ins_h],e.head[e.ins_h]=e.strstart),n!==0&&e.strstart-n<=e.w_size-262&&(e.match_length=jg(e,n)),e.match_length>=3){if(r=ng(e,e.strstart-e.match_start,e.match_length-3),e.lookahead-=e.match_length,e.match_length<=e.max_lazy_match&&e.lookahead>=3){e.match_length--;do e.strstart++,e.ins_h=Eg(e,e.ins_h,e.window[e.strstart+3-1]),n=e.prev[e.strstart&e.w_mask]=e.head[e.ins_h],e.head[e.ins_h]=e.strstart;while(--e.match_length!==0);e.strstart++}else e.strstart+=e.match_length,e.match_length=0,e.ins_h=e.window[e.strstart],e.ins_h=Eg(e,e.ins_h,e.window[e.strstart+1])}else r=ng(e,0,e.window[e.strstart]),e.lookahead--,e.strstart++;if(r&&(Og(e,!1),e.strm.avail_out===0))return 1}return e.insert=e.strstart<2?e.strstart:2,t===sg?(Og(e,!0),e.strm.avail_out===0?3:4):e.sym_next&&(Og(e,!1),e.strm.avail_out===0)?1:2},Fg=(e,t)=>{let n,r,i;for(;;){if(e.lookahead<262){if(Mg(e),e.lookahead<262&&t===ig)return 1;if(e.lookahead===0)break}if(n=0,e.lookahead>=3&&(e.ins_h=Eg(e,e.ins_h,e.window[e.strstart+3-1]),n=e.prev[e.strstart&e.w_mask]=e.head[e.ins_h],e.head[e.ins_h]=e.strstart),e.prev_length=e.match_length,e.prev_match=e.match_start,e.match_length=2,n!==0&&e.prev_length<e.max_lazy_match&&e.strstart-n<=e.w_size-262&&(e.match_length=jg(e,n),e.match_length<=5&&(e.strategy===hg||e.match_length===3&&e.strstart-e.match_start>4096)&&(e.match_length=2)),e.prev_length>=3&&e.match_length<=e.prev_length){i=e.strstart+e.lookahead-3,r=ng(e,e.strstart-1-e.prev_match,e.prev_length-3),e.lookahead-=e.prev_length-1,e.prev_length-=2;do++e.strstart<=i&&(e.ins_h=Eg(e,e.ins_h,e.window[e.strstart+3-1]),n=e.prev[e.strstart&e.w_mask]=e.head[e.ins_h],e.head[e.ins_h]=e.strstart);while(--e.prev_length!==0);if(e.match_available=0,e.match_length=2,e.strstart++,r&&(Og(e,!1),e.strm.avail_out===0))return 1}else if(e.match_available){if(r=ng(e,0,e.window[e.strstart-1]),r&&Og(e,!1),e.strstart++,e.lookahead--,e.strm.avail_out===0)return 1}else e.match_available=1,e.strstart++,e.lookahead--}return e.match_available&&=(r=ng(e,0,e.window[e.strstart-1]),0),e.insert=e.strstart<2?e.strstart:2,t===sg?(Og(e,!0),e.strm.avail_out===0?3:4):e.sym_next&&(Og(e,!1),e.strm.avail_out===0)?1:2},Ig=(e,t)=>{let n,r,i,a,o=e.window;for(;;){if(e.lookahead<=258){if(Mg(e),e.lookahead<=258&&t===ig)return 1;if(e.lookahead===0)break}if(e.match_length=0,e.lookahead>=3&&e.strstart>0&&(i=e.strstart-1,r=o[i],r===o[++i]&&r===o[++i]&&r===o[++i])){a=e.strstart+258;do;while(r===o[++i]&&r===o[++i]&&r===o[++i]&&r===o[++i]&&r===o[++i]&&r===o[++i]&&r===o[++i]&&r===o[++i]&&i<a);e.match_length=258-(a-i),e.match_length>e.lookahead&&(e.match_length=e.lookahead)}if(e.match_length>=3?(n=ng(e,1,e.match_length-3),e.lookahead-=e.match_length,e.strstart+=e.match_length,e.match_length=0):(n=ng(e,0,e.window[e.strstart]),e.lookahead--,e.strstart++),n&&(Og(e,!1),e.strm.avail_out===0))return 1}return e.insert=0,t===sg?(Og(e,!0),e.strm.avail_out===0?3:4):e.sym_next&&(Og(e,!1),e.strm.avail_out===0)?1:2},Lg=(e,t)=>{let n;for(;;){if(e.lookahead===0&&(Mg(e),e.lookahead===0)){if(t===ig)return 1;break}if(e.match_length=0,n=ng(e,0,e.window[e.strstart]),e.lookahead--,e.strstart++,n&&(Og(e,!1),e.strm.avail_out===0))return 1}return e.insert=0,t===sg?(Og(e,!0),e.strm.avail_out===0?3:4):e.sym_next&&(Og(e,!1),e.strm.avail_out===0)?1:2};function Rg(e,t,n,r,i){this.good_length=e,this.max_lazy=t,this.nice_length=n,this.max_chain=r,this.func=i}let zg=[new Rg(0,0,0,0,Ng),new Rg(4,4,8,4,Pg),new Rg(4,5,16,8,Pg),new Rg(4,6,32,32,Pg),new Rg(4,4,16,16,Fg),new Rg(8,16,32,32,Fg),new Rg(8,16,128,128,Fg),new Rg(8,32,128,256,Fg),new Rg(32,128,258,1024,Fg),new Rg(32,258,258,4096,Fg)],Bg=e=>{e.window_size=2*e.w_size,wg(e.head),e.max_lazy_match=zg[e.level].max_lazy,e.good_match=zg[e.level].good_length,e.nice_match=zg[e.level].nice_length,e.max_chain_length=zg[e.level].max_chain,e.strstart=0,e.block_start=0,e.lookahead=0,e.insert=0,e.match_length=e.prev_length=2,e.match_available=0,e.ins_h=0};function Vg(){this.strm=null,this.status=0,this.pending_buf=null,this.pending_buf_size=0,this.pending_out=0,this.pending=0,this.wrap=0,this.gzhead=null,this.gzindex=0,this.method=xg,this.last_flush=-1,this.w_size=0,this.w_bits=0,this.w_mask=0,this.window=null,this.window_size=0,this.prev=null,this.head=null,this.ins_h=0,this.hash_size=0,this.hash_bits=0,this.hash_mask=0,this.hash_shift=0,this.block_start=0,this.match_length=0,this.prev_match=0,this.match_available=0,this.strstart=0,this.match_start=0,this.lookahead=0,this.prev_length=0,this.max_chain_length=0,this.max_lazy_match=0,this.level=0,this.strategy=0,this.good_match=0,this.nice_match=0,this.dyn_ltree=new Uint16Array(1146),this.dyn_dtree=new Uint16Array(122),this.bl_tree=new Uint16Array(78),wg(this.dyn_ltree),wg(this.dyn_dtree),wg(this.bl_tree),this.l_desc=null,this.d_desc=null,this.bl_desc=null,this.bl_count=new Uint16Array(16),this.heap=new Uint16Array(573),wg(this.heap),this.heap_len=0,this.heap_max=0,this.depth=new Uint16Array(573),wg(this.depth),this.sym_buf=0,this.lit_bufsize=0,this.sym_next=0,this.sym_end=0,this.opt_len=0,this.static_len=0,this.matches=0,this.insert=0,this.bi_buf=0,this.bi_valid=0}let Hg=e=>{if(!e)return 1;let t=e.state;return+(!t||t.strm!==e||t.status!==42&&t.status!==57&&t.status!==69&&t.status!==73&&t.status!==91&&t.status!==103&&t.status!==113&&t.status!==666)},Ug=e=>{if(Hg(e))return Sg(e,dg);e.total_in=e.total_out=0,e.data_type=bg;let t=e.state;return t.pending=0,t.pending_out=0,t.wrap<0&&(t.wrap=-t.wrap),t.status=t.wrap===2?57:t.wrap?42:113,e.adler=t.wrap===2?0:1,t.last_flush=-2,$h(t),lg},Wg=e=>{let t=Ug(e);return t===lg&&Bg(e.state),t},Gg=(e,t)=>Hg(e)||e.state.wrap!==2?dg:(e.state.gzhead=t,lg),Kg=(e,t,n,r,i,a)=>{if(!e)return dg;let o=1;if(t===mg&&(t=6),r<0?(o=0,r=-r):r>15&&(o=2,r-=16),i<1||i>9||n!==xg||r<8||r>15||t<0||t>9||a<0||a>vg||r===8&&o!==1)return Sg(e,dg);r===8&&(r=9);let s=new Vg;return e.state=s,s.strm=e,s.status=42,s.wrap=o,s.gzhead=null,s.w_bits=r,s.w_size=1<<s.w_bits,s.w_mask=s.w_size-1,s.hash_bits=i+7,s.hash_size=1<<s.hash_bits,s.hash_mask=s.hash_size-1,s.hash_shift=~~((s.hash_bits+3-1)/3),s.window=new Uint8Array(s.w_size*2),s.head=new Uint16Array(s.hash_size),s.prev=new Uint16Array(s.w_size),s.lit_bufsize=1<<i+6,s.pending_buf_size=s.lit_bufsize*4,s.pending_buf=new Uint8Array(s.pending_buf_size),s.sym_buf=s.lit_bufsize,s.sym_end=(s.lit_bufsize-1)*3,s.level=t,s.strategy=a,s.method=n,Wg(e)};var qg={deflateInit:(e,t)=>Kg(e,t,xg,15,8,yg),deflateInit2:Kg,deflateReset:Wg,deflateResetKeep:Ug,deflateSetHeader:Gg,deflate:(e,t)=>{if(Hg(e)||t>cg||t<0)return e?Sg(e,dg):dg;let n=e.state;if(!e.output||e.avail_in!==0&&!e.input||n.status===666&&t!==sg)return Sg(e,e.avail_out===0?pg:dg);let r=n.last_flush;if(n.last_flush=t,n.pending!==0){if(Dg(e),e.avail_out===0)return n.last_flush=-1,lg}else if(e.avail_in===0&&Cg(t)<=Cg(r)&&t!==sg)return Sg(e,pg);if(n.status===666&&e.avail_in!==0)return Sg(e,pg);if(n.status===42&&n.wrap===0&&(n.status=113),n.status===42){let t=xg+(n.w_bits-8<<4)<<8,r=-1;if(r=n.strategy>=gg||n.level<2?0:n.level<6?1:n.level===6?2:3,t|=r<<6,n.strstart!==0&&(t|=32),t+=31-t%31,kg(n,t),n.strstart!==0&&(kg(n,e.adler>>>16),kg(n,e.adler&65535)),e.adler=1,n.status=113,Dg(e),n.pending!==0)return n.last_flush=-1,lg}if(n.status===57){if(e.adler=0,Y(n,31),Y(n,139),Y(n,8),n.gzhead)Y(n,+!!n.gzhead.text+(n.gzhead.hcrc?2:0)+(n.gzhead.extra?4:0)+(n.gzhead.name?8:0)+(n.gzhead.comment?16:0)),Y(n,n.gzhead.time&255),Y(n,n.gzhead.time>>8&255),Y(n,n.gzhead.time>>16&255),Y(n,n.gzhead.time>>24&255),Y(n,n.level===9?2:n.strategy>=gg||n.level<2?4:0),Y(n,n.gzhead.os&255),n.gzhead.extra&&n.gzhead.extra.length&&(Y(n,n.gzhead.extra.length&255),Y(n,n.gzhead.extra.length>>8&255)),n.gzhead.hcrc&&(e.adler=Xh(e.adler,n.pending_buf,n.pending,0)),n.gzindex=0,n.status=69;else if(Y(n,0),Y(n,0),Y(n,0),Y(n,0),Y(n,0),Y(n,n.level===9?2:n.strategy>=gg||n.level<2?4:0),Y(n,3),n.status=113,Dg(e),n.pending!==0)return n.last_flush=-1,lg}if(n.status===69){if(n.gzhead.extra){let t=n.pending,r=(n.gzhead.extra.length&65535)-n.gzindex;for(;n.pending+r>n.pending_buf_size;){let i=n.pending_buf_size-n.pending;if(n.pending_buf.set(n.gzhead.extra.subarray(n.gzindex,n.gzindex+i),n.pending),n.pending=n.pending_buf_size,n.gzhead.hcrc&&n.pending>t&&(e.adler=Xh(e.adler,n.pending_buf,n.pending-t,t)),n.gzindex+=i,Dg(e),n.pending!==0)return n.last_flush=-1,lg;t=0,r-=i}let i=new Uint8Array(n.gzhead.extra);n.pending_buf.set(i.subarray(n.gzindex,n.gzindex+r),n.pending),n.pending+=r,n.gzhead.hcrc&&n.pending>t&&(e.adler=Xh(e.adler,n.pending_buf,n.pending-t,t)),n.gzindex=0}n.status=73}if(n.status===73){if(n.gzhead.name){let t=n.pending,r;do{if(n.pending===n.pending_buf_size){if(n.gzhead.hcrc&&n.pending>t&&(e.adler=Xh(e.adler,n.pending_buf,n.pending-t,t)),Dg(e),n.pending!==0)return n.last_flush=-1,lg;t=0}r=n.gzindex<n.gzhead.name.length?n.gzhead.name.charCodeAt(n.gzindex++)&255:0,Y(n,r)}while(r!==0);n.gzhead.hcrc&&n.pending>t&&(e.adler=Xh(e.adler,n.pending_buf,n.pending-t,t)),n.gzindex=0}n.status=91}if(n.status===91){if(n.gzhead.comment){let t=n.pending,r;do{if(n.pending===n.pending_buf_size){if(n.gzhead.hcrc&&n.pending>t&&(e.adler=Xh(e.adler,n.pending_buf,n.pending-t,t)),Dg(e),n.pending!==0)return n.last_flush=-1,lg;t=0}r=n.gzindex<n.gzhead.comment.length?n.gzhead.comment.charCodeAt(n.gzindex++)&255:0,Y(n,r)}while(r!==0);n.gzhead.hcrc&&n.pending>t&&(e.adler=Xh(e.adler,n.pending_buf,n.pending-t,t))}n.status=103}if(n.status===103){if(n.gzhead.hcrc){if(n.pending+2>n.pending_buf_size&&(Dg(e),n.pending!==0))return n.last_flush=-1,lg;Y(n,e.adler&255),Y(n,e.adler>>8&255),e.adler=0}if(n.status=113,Dg(e),n.pending!==0)return n.last_flush=-1,lg}if(e.avail_in!==0||n.lookahead!==0||t!==ig&&n.status!==666){let r=n.level===0?Ng(n,t):n.strategy===gg?Lg(n,t):n.strategy===_g?Ig(n,t):zg[n.level].func(n,t);if((r===3||r===4)&&(n.status=666),r===1||r===3)return e.avail_out===0&&(n.last_flush=-1),lg;if(r===2&&(t===ag?rg(n):t!==cg&&(eg(n,0,0,!1),t===og&&(wg(n.head),n.lookahead===0&&(n.strstart=0,n.block_start=0,n.insert=0))),Dg(e),e.avail_out===0))return n.last_flush=-1,lg}return t===sg?n.wrap<=0?ug:(n.wrap===2?(Y(n,e.adler&255),Y(n,e.adler>>8&255),Y(n,e.adler>>16&255),Y(n,e.adler>>24&255),Y(n,e.total_in&255),Y(n,e.total_in>>8&255),Y(n,e.total_in>>16&255),Y(n,e.total_in>>24&255)):(kg(n,e.adler>>>16),kg(n,e.adler&65535)),Dg(e),n.wrap>0&&(n.wrap=-n.wrap),n.pending===0?ug:lg):lg},deflateEnd:e=>{if(Hg(e))return dg;let t=e.state.status;return e.state=null,t===113?Sg(e,fg):lg},deflateSetDictionary:(e,t)=>{let n=t.length;if(Hg(e))return dg;let r=e.state,i=r.wrap;if(i===2||i===1&&r.status!==42||r.lookahead)return dg;if(i===1&&(e.adler=Jh(e.adler,t,n,0)),r.wrap=0,n>=r.w_size){i===0&&(wg(r.head),r.strstart=0,r.block_start=0,r.insert=0);let e=new Uint8Array(r.w_size);e.set(t.subarray(n-r.w_size,n),0),t=e,n=r.w_size}let a=e.avail_in,o=e.next_in,s=e.input;for(e.avail_in=n,e.next_in=0,e.input=t,Mg(r);r.lookahead>=3;){let e=r.strstart,t=r.lookahead-2;do r.ins_h=Eg(r,r.ins_h,r.window[e+3-1]),r.prev[e&r.w_mask]=r.head[r.ins_h],r.head[r.ins_h]=e,e++;while(--t);r.strstart=e,r.lookahead=2,Mg(r)}return r.strstart+=r.lookahead,r.block_start=r.strstart,r.insert=r.lookahead,r.lookahead=0,r.match_length=r.prev_length=2,r.match_available=0,e.next_in=o,e.input=s,e.avail_in=a,r.wrap=i,lg},deflateInfo:`pako deflate (from Nodeca project)`};let Jg=(e,t)=>Object.prototype.hasOwnProperty.call(e,t);var Yg={assign:function(e){let t=Array.prototype.slice.call(arguments,1);for(;t.length;){let n=t.shift();if(n){if(typeof n!=`object`)throw TypeError(n+`must be non-object`);for(let t in n)Jg(n,t)&&(e[t]=n[t])}}return e},flattenChunks:e=>{let t=0;for(let n=0,r=e.length;n<r;n++)t+=e[n].length;let n=new Uint8Array(t);for(let t=0,r=0,i=e.length;t<i;t++){let i=e[t];n.set(i,r),r+=i.length}return n}};let Xg=!0;try{String.fromCharCode.apply(null,new Uint8Array(1))}catch{Xg=!1}let Zg=new Uint8Array(256);for(let e=0;e<256;e++)Zg[e]=e>=252?6:e>=248?5:e>=240?4:e>=224?3:e>=192?2:1;Zg[254]=Zg[254]=1;var Qg=e=>{if(typeof TextEncoder==`function`&&TextEncoder.prototype.encode)return new TextEncoder().encode(e);let t,n,r,i,a,o=e.length,s=0;for(i=0;i<o;i++)n=e.charCodeAt(i),(n&64512)==55296&&i+1<o&&(r=e.charCodeAt(i+1),(r&64512)==56320&&(n=65536+(n-55296<<10)+(r-56320),i++)),s+=n<128?1:n<2048?2:n<65536?3:4;for(t=new Uint8Array(s),a=0,i=0;a<s;i++)n=e.charCodeAt(i),(n&64512)==55296&&i+1<o&&(r=e.charCodeAt(i+1),(r&64512)==56320&&(n=65536+(n-55296<<10)+(r-56320),i++)),n<128?t[a++]=n:n<2048?(t[a++]=192|n>>>6,t[a++]=128|n&63):n<65536?(t[a++]=224|n>>>12,t[a++]=128|n>>>6&63,t[a++]=128|n&63):(t[a++]=240|n>>>18,t[a++]=128|n>>>12&63,t[a++]=128|n>>>6&63,t[a++]=128|n&63);return t};let $g=(e,t)=>{if(t<65534&&e.subarray&&Xg)return String.fromCharCode.apply(null,e.length===t?e:e.subarray(0,t));let n=``;for(let r=0;r<t;r++)n+=String.fromCharCode(e[r]);return n};var e_={string2buf:Qg,buf2string:(e,t)=>{let n=t||e.length;if(typeof TextDecoder==`function`&&TextDecoder.prototype.decode)return new TextDecoder().decode(e.subarray(0,t));let r,i,a=Array(n*2);for(i=0,r=0;r<n;){let t=e[r++];if(t<128){a[i++]=t;continue}let o=Zg[t];if(o>4){a[i++]=65533,r+=o-1;continue}for(t&=o===2?31:o===3?15:7;o>1&&r<n;)t=t<<6|e[r++]&63,o--;if(o>1){a[i++]=65533;continue}t<65536?a[i++]=t:(t-=65536,a[i++]=55296|t>>10&1023,a[i++]=56320|t&1023)}return $g(a,i)},utf8border:(e,t)=>{t||=e.length,t>e.length&&(t=e.length);let n=t-1;for(;n>=0&&(e[n]&192)==128;)n--;return n<0||n===0?t:n+Zg[e[n]]>t?n:t}};function t_(){this.input=null,this.next_in=0,this.avail_in=0,this.total_in=0,this.output=null,this.next_out=0,this.avail_out=0,this.total_out=0,this.msg=``,this.state=null,this.data_type=2,this.adler=0}var n_=t_;let r_=Object.prototype.toString,{Z_NO_FLUSH:i_,Z_SYNC_FLUSH:a_,Z_FULL_FLUSH:o_,Z_FINISH:s_,Z_OK:c_,Z_STREAM_END:l_,Z_DEFAULT_COMPRESSION:u_,Z_DEFAULT_STRATEGY:d_,Z_DEFLATED:f_}=Qh;function p_(e){this.options=Yg.assign({level:u_,method:f_,chunkSize:16384,windowBits:15,memLevel:8,strategy:d_},e||{});let t=this.options;t.raw&&t.windowBits>0?t.windowBits=-t.windowBits:t.gzip&&t.windowBits>0&&t.windowBits<16&&(t.windowBits+=16),this.err=0,this.msg=``,this.ended=!1,this.chunks=[],this.strm=new n_,this.strm.avail_out=0;let n=qg.deflateInit2(this.strm,t.level,t.method,t.windowBits,t.memLevel,t.strategy);if(n!==c_)throw Error(Zh[n]);if(t.header&&qg.deflateSetHeader(this.strm,t.header),t.dictionary){let e;if(e=typeof t.dictionary==`string`?e_.string2buf(t.dictionary):r_.call(t.dictionary)===`[object ArrayBuffer]`?new Uint8Array(t.dictionary):t.dictionary,n=qg.deflateSetDictionary(this.strm,e),n!==c_)throw Error(Zh[n]);this._dict_set=!0}}p_.prototype.push=function(e,t){let n=this.strm,r=this.options.chunkSize,i,a;if(this.ended)return!1;for(a=t===~~t?t:t===!0?s_:i_,n.input=typeof e==`string`?e_.string2buf(e):r_.call(e)===`[object ArrayBuffer]`?new Uint8Array(e):e,n.next_in=0,n.avail_in=n.input.length;;){if(n.avail_out===0&&(n.output=new Uint8Array(r),n.next_out=0,n.avail_out=r),(a===a_||a===o_)&&n.avail_out<=6){this.onData(n.output.subarray(0,n.next_out)),n.avail_out=0;continue}if(i=qg.deflate(n,a),i===l_)return n.next_out>0&&this.onData(n.output.subarray(0,n.next_out)),i=qg.deflateEnd(this.strm),this.onEnd(i),this.ended=!0,i===c_;if(n.avail_out===0){this.onData(n.output);continue}if(a>0&&n.next_out>0){this.onData(n.output.subarray(0,n.next_out)),n.avail_out=0;continue}if(n.avail_in===0)break}return!0},p_.prototype.onData=function(e){this.chunks.push(e)},p_.prototype.onEnd=function(e){e===c_&&(this.result=Yg.flattenChunks(this.chunks)),this.chunks=[],this.err=e,this.msg=this.strm.msg};function m_(e,t){let n=new p_(t);if(n.push(e,!0),n.err)throw n.msg||Zh[n.err];return n.result}function h_(e,t){return t||={},t.raw=!0,m_(e,t)}function g_(e,t){return t||={},t.gzip=!0,m_(e,t)}var __={Deflate:p_,deflate:m_,deflateRaw:h_,gzip:g_,constants:Qh};let v_=16209;var y_=function(e,t){let n,r,i,a,o,s,c,l,u,d,f,p,m,h,g,_,v,y,b,x,S,C,w,T,E=e.state;n=e.next_in,w=e.input,r=n+(e.avail_in-5),i=e.next_out,T=e.output,a=i-(t-e.avail_out),o=i+(e.avail_out-257),s=E.dmax,c=E.wsize,l=E.whave,u=E.wnext,d=E.window,f=E.hold,p=E.bits,m=E.lencode,h=E.distcode,g=(1<<E.lenbits)-1,_=(1<<E.distbits)-1;top:do{p<15&&(f+=w[n++]<<p,p+=8,f+=w[n++]<<p,p+=8),v=m[f&g];dolen:for(;;){if(y=v>>>24,f>>>=y,p-=y,y=v>>>16&255,y===0)T[i++]=v&65535;else if(y&16){b=v&65535,y&=15,y&&(p<y&&(f+=w[n++]<<p,p+=8),b+=f&(1<<y)-1,f>>>=y,p-=y),p<15&&(f+=w[n++]<<p,p+=8,f+=w[n++]<<p,p+=8),v=h[f&_];dodist:for(;;){if(y=v>>>24,f>>>=y,p-=y,y=v>>>16&255,y&16){if(x=v&65535,y&=15,p<y&&(f+=w[n++]<<p,p+=8,p<y&&(f+=w[n++]<<p,p+=8)),x+=f&(1<<y)-1,x>s){e.msg=`invalid distance too far back`,E.mode=v_;break top}if(f>>>=y,p-=y,y=i-a,x>y){if(y=x-y,y>l&&E.sane){e.msg=`invalid distance too far back`,E.mode=v_;break top}if(S=0,C=d,u===0){if(S+=c-y,y<b){b-=y;do T[i++]=d[S++];while(--y);S=i-x,C=T}}else if(u<y){if(S+=c+u-y,y-=u,y<b){b-=y;do T[i++]=d[S++];while(--y);if(S=0,u<b){y=u,b-=y;do T[i++]=d[S++];while(--y);S=i-x,C=T}}}else if(S+=u-y,y<b){b-=y;do T[i++]=d[S++];while(--y);S=i-x,C=T}for(;b>2;)T[i++]=C[S++],T[i++]=C[S++],T[i++]=C[S++],b-=3;b&&(T[i++]=C[S++],b>1&&(T[i++]=C[S++]))}else{S=i-x;do T[i++]=T[S++],T[i++]=T[S++],T[i++]=T[S++],b-=3;while(b>2);b&&(T[i++]=T[S++],b>1&&(T[i++]=T[S++]))}}else if(y&64){e.msg=`invalid distance code`,E.mode=v_;break top}else{v=h[(v&65535)+(f&(1<<y)-1)];continue dodist}break}}else if(!(y&64)){v=m[(v&65535)+(f&(1<<y)-1)];continue dolen}else if(y&32){E.mode=16191;break top}else{e.msg=`invalid literal/length code`,E.mode=v_;break top}break}}while(n<r&&i<o);b=p>>3,n-=b,p-=b<<3,f&=(1<<p)-1,e.next_in=n,e.next_out=i,e.avail_in=n<r?5+(r-n):5-(n-r),e.avail_out=i<o?257+(o-i):257-(i-o),E.hold=f,E.bits=p};let b_=new Uint16Array([3,4,5,6,7,8,9,10,11,13,15,17,19,23,27,31,35,43,51,59,67,83,99,115,131,163,195,227,258,0,0]),x_=new Uint8Array([16,16,16,16,16,16,16,16,17,17,17,17,18,18,18,18,19,19,19,19,20,20,20,20,21,21,21,21,16,72,78]),S_=new Uint16Array([1,2,3,4,5,7,9,13,17,25,33,49,65,97,129,193,257,385,513,769,1025,1537,2049,3073,4097,6145,8193,12289,16385,24577,0,0]),C_=new Uint8Array([16,16,16,16,17,17,18,18,19,19,20,20,21,21,22,22,23,23,24,24,25,25,26,26,27,27,28,28,29,29,64,64]);var w_=(e,t,n,r,i,a,o,s)=>{let c=s.bits,l=0,u=0,d=0,f=0,p=0,m=0,h=0,g=0,_=0,v=0,y,b,x,S,C,w=null,T,E=new Uint16Array(16),D=new Uint16Array(16),O=null,k,A,j;for(l=0;l<=15;l++)E[l]=0;for(u=0;u<r;u++)E[t[n+u]]++;for(p=c,f=15;f>=1&&E[f]===0;f--);if(p>f&&(p=f),f===0)return i[a++]=20971520,i[a++]=20971520,s.bits=1,0;for(d=1;d<f&&E[d]===0;d++);for(p<d&&(p=d),g=1,l=1;l<=15;l++)if(g<<=1,g-=E[l],g<0)return-1;if(g>0&&(e===0||f!==1))return-1;for(D[1]=0,l=1;l<15;l++)D[l+1]=D[l]+E[l];for(u=0;u<r;u++)t[n+u]!==0&&(o[D[t[n+u]]++]=u);if(e===0?(w=O=o,T=20):e===1?(w=b_,O=x_,T=257):(w=S_,O=C_,T=0),v=0,u=0,l=d,C=a,m=p,h=0,x=-1,_=1<<p,S=_-1,e===1&&_>852||e===2&&_>592)return 1;for(;;){k=l-h,o[u]+1<T?(A=0,j=o[u]):o[u]>=T?(A=O[o[u]-T],j=w[o[u]-T]):(A=96,j=0),y=1<<l-h,b=1<<m,d=b;do b-=y,i[C+(v>>h)+b]=k<<24|A<<16|j|0;while(b!==0);for(y=1<<l-1;v&y;)y>>=1;if(y===0?v=0:(v&=y-1,v+=y),u++,--E[l]===0){if(l===f)break;l=t[n+o[u]]}if(l>p&&(v&S)!==x){for(h===0&&(h=p),C+=d,m=l-h,g=1<<m;m+h<f&&(g-=E[m+h],!(g<=0));)m++,g<<=1;if(_+=1<<m,e===1&&_>852||e===2&&_>592)return 1;x=v&S,i[x]=p<<24|m<<16|C-a|0}}return v!==0&&(i[C+v]=l-h<<24|4194304),s.bits=p,0};let{Z_FINISH:T_,Z_BLOCK:E_,Z_TREES:D_,Z_OK:O_,Z_STREAM_END:k_,Z_NEED_DICT:A_,Z_STREAM_ERROR:j_,Z_DATA_ERROR:M_,Z_MEM_ERROR:N_,Z_BUF_ERROR:P_,Z_DEFLATED:F_}=Qh,I_=16180,L_=16181,R_=16182,z_=16183,B_=16184,V_=16185,H_=16186,U_=16187,W_=16188,G_=16189,K_=16190,q_=16191,J_=16192,Y_=16193,X_=16194,Z_=16195,Q_=16196,$_=16197,ev=16198,tv=16199,nv=16200,rv=16201,iv=16202,av=16203,ov=16204,sv=16205,cv=16206,lv=16207,uv=16208,dv=16209,fv=16210,pv=16211,mv=e=>(e>>>24&255)+(e>>>8&65280)+((e&65280)<<8)+((e&255)<<24);function hv(){this.strm=null,this.mode=0,this.last=!1,this.wrap=0,this.havedict=!1,this.flags=0,this.dmax=0,this.check=0,this.total=0,this.head=null,this.wbits=0,this.wsize=0,this.whave=0,this.wnext=0,this.window=null,this.hold=0,this.bits=0,this.length=0,this.offset=0,this.extra=0,this.lencode=null,this.distcode=null,this.lenbits=0,this.distbits=0,this.ncode=0,this.nlen=0,this.ndist=0,this.have=0,this.next=null,this.lens=new Uint16Array(320),this.work=new Uint16Array(288),this.lendyn=null,this.distdyn=null,this.sane=0,this.back=0,this.was=0}let gv=e=>{if(!e)return 1;let t=e.state;return+(!t||t.strm!==e||t.mode<I_||t.mode>pv)},_v=e=>{if(gv(e))return j_;let t=e.state;return e.total_in=e.total_out=t.total=0,e.msg=``,t.wrap&&(e.adler=t.wrap&1),t.mode=I_,t.last=0,t.havedict=0,t.flags=-1,t.dmax=32768,t.head=null,t.hold=0,t.bits=0,t.lencode=t.lendyn=new Int32Array(852),t.distcode=t.distdyn=new Int32Array(592),t.sane=1,t.back=-1,O_},vv=e=>{if(gv(e))return j_;let t=e.state;return t.wsize=0,t.whave=0,t.wnext=0,_v(e)},yv=(e,t)=>{let n;if(gv(e))return j_;let r=e.state;return t<0?(n=0,t=-t):(n=(t>>4)+5,t<48&&(t&=15)),t&&(t<8||t>15)?j_:(r.window!==null&&r.wbits!==t&&(r.window=null),r.wrap=n,r.wbits=t,vv(e))},bv=(e,t)=>{if(!e)return j_;let n=new hv;e.state=n,n.strm=e,n.window=null,n.mode=I_;let r=yv(e,t);return r!==O_&&(e.state=null),r},xv=e=>bv(e,15),Sv=!0,Cv,wv,Tv=e=>{if(Sv){Cv=new Int32Array(512),wv=new Int32Array(32);let t=0;for(;t<144;)e.lens[t++]=8;for(;t<256;)e.lens[t++]=9;for(;t<280;)e.lens[t++]=7;for(;t<288;)e.lens[t++]=8;for(w_(1,e.lens,0,288,Cv,0,e.work,{bits:9}),t=0;t<32;)e.lens[t++]=5;w_(2,e.lens,0,32,wv,0,e.work,{bits:5}),Sv=!1}e.lencode=Cv,e.lenbits=9,e.distcode=wv,e.distbits=5},Ev=(e,t,n,r)=>{let i,a=e.state;return a.window===null&&(a.wsize=1<<a.wbits,a.wnext=0,a.whave=0,a.window=new Uint8Array(a.wsize)),r>=a.wsize?(a.window.set(t.subarray(n-a.wsize,n),0),a.wnext=0,a.whave=a.wsize):(i=a.wsize-a.wnext,i>r&&(i=r),a.window.set(t.subarray(n-r,n-r+i),a.wnext),r-=i,r?(a.window.set(t.subarray(n-r,n),0),a.wnext=r,a.whave=a.wsize):(a.wnext+=i,a.wnext===a.wsize&&(a.wnext=0),a.whave<a.wsize&&(a.whave+=i))),0};var Dv={inflateReset:vv,inflateReset2:yv,inflateResetKeep:_v,inflateInit:xv,inflateInit2:bv,inflate:(e,t)=>{let n,r,i,a,o,s,c,l,u,d,f,p,m,h,g=0,_,v,y,b,x,S,C,w,T=new Uint8Array(4),E,D,O=new Uint8Array([16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15]);if(gv(e)||!e.output||!e.input&&e.avail_in!==0)return j_;n=e.state,n.mode===q_&&(n.mode=J_),o=e.next_out,i=e.output,c=e.avail_out,a=e.next_in,r=e.input,s=e.avail_in,l=n.hold,u=n.bits,d=s,f=c,w=O_;inf_leave:for(;;)switch(n.mode){case I_:if(n.wrap===0){n.mode=J_;break}for(;u<16;){if(s===0)break inf_leave;s--,l+=r[a++]<<u,u+=8}if(n.wrap&2&&l===35615){n.wbits===0&&(n.wbits=15),n.check=0,T[0]=l&255,T[1]=l>>>8&255,n.check=Xh(n.check,T,2,0),l=0,u=0,n.mode=L_;break}if(n.head&&(n.head.done=!1),!(n.wrap&1)||(((l&255)<<8)+(l>>8))%31){e.msg=`incorrect header check`,n.mode=dv;break}if((l&15)!==F_){e.msg=`unknown compression method`,n.mode=dv;break}if(l>>>=4,u-=4,C=(l&15)+8,n.wbits===0&&(n.wbits=C),C>15||C>n.wbits){e.msg=`invalid window size`,n.mode=dv;break}n.dmax=1<<n.wbits,n.flags=0,e.adler=n.check=1,n.mode=l&512?G_:q_,l=0,u=0;break;case L_:for(;u<16;){if(s===0)break inf_leave;s--,l+=r[a++]<<u,u+=8}if(n.flags=l,(n.flags&255)!==F_){e.msg=`unknown compression method`,n.mode=dv;break}if(n.flags&57344){e.msg=`unknown header flags set`,n.mode=dv;break}n.head&&(n.head.text=l>>8&1),n.flags&512&&n.wrap&4&&(T[0]=l&255,T[1]=l>>>8&255,n.check=Xh(n.check,T,2,0)),l=0,u=0,n.mode=R_;case R_:for(;u<32;){if(s===0)break inf_leave;s--,l+=r[a++]<<u,u+=8}n.head&&(n.head.time=l),n.flags&512&&n.wrap&4&&(T[0]=l&255,T[1]=l>>>8&255,T[2]=l>>>16&255,T[3]=l>>>24&255,n.check=Xh(n.check,T,4,0)),l=0,u=0,n.mode=z_;case z_:for(;u<16;){if(s===0)break inf_leave;s--,l+=r[a++]<<u,u+=8}n.head&&(n.head.xflags=l&255,n.head.os=l>>8),n.flags&512&&n.wrap&4&&(T[0]=l&255,T[1]=l>>>8&255,n.check=Xh(n.check,T,2,0)),l=0,u=0,n.mode=B_;case B_:if(n.flags&1024){for(;u<16;){if(s===0)break inf_leave;s--,l+=r[a++]<<u,u+=8}n.length=l,n.head&&(n.head.extra_len=l),n.flags&512&&n.wrap&4&&(T[0]=l&255,T[1]=l>>>8&255,n.check=Xh(n.check,T,2,0)),l=0,u=0}else n.head&&(n.head.extra=null);n.mode=V_;case V_:if(n.flags&1024&&(p=n.length,p>s&&(p=s),p&&(n.head&&(C=n.head.extra_len-n.length,n.head.extra||(n.head.extra=new Uint8Array(n.head.extra_len)),n.head.extra.set(r.subarray(a,a+p),C)),n.flags&512&&n.wrap&4&&(n.check=Xh(n.check,r,p,a)),s-=p,a+=p,n.length-=p),n.length))break inf_leave;n.length=0,n.mode=H_;case H_:if(n.flags&2048){if(s===0)break inf_leave;p=0;do C=r[a+p++],n.head&&C&&n.length<65536&&(n.head.name+=String.fromCharCode(C));while(C&&p<s);if(n.flags&512&&n.wrap&4&&(n.check=Xh(n.check,r,p,a)),s-=p,a+=p,C)break inf_leave}else n.head&&(n.head.name=null);n.length=0,n.mode=U_;case U_:if(n.flags&4096){if(s===0)break inf_leave;p=0;do C=r[a+p++],n.head&&C&&n.length<65536&&(n.head.comment+=String.fromCharCode(C));while(C&&p<s);if(n.flags&512&&n.wrap&4&&(n.check=Xh(n.check,r,p,a)),s-=p,a+=p,C)break inf_leave}else n.head&&(n.head.comment=null);n.mode=W_;case W_:if(n.flags&512){for(;u<16;){if(s===0)break inf_leave;s--,l+=r[a++]<<u,u+=8}if(n.wrap&4&&l!==(n.check&65535)){e.msg=`header crc mismatch`,n.mode=dv;break}l=0,u=0}n.head&&(n.head.hcrc=n.flags>>9&1,n.head.done=!0),e.adler=n.check=0,n.mode=q_;break;case G_:for(;u<32;){if(s===0)break inf_leave;s--,l+=r[a++]<<u,u+=8}e.adler=n.check=mv(l),l=0,u=0,n.mode=K_;case K_:if(n.havedict===0)return e.next_out=o,e.avail_out=c,e.next_in=a,e.avail_in=s,n.hold=l,n.bits=u,A_;e.adler=n.check=1,n.mode=q_;case q_:if(t===E_||t===D_)break inf_leave;case J_:if(n.last){l>>>=u&7,u-=u&7,n.mode=cv;break}for(;u<3;){if(s===0)break inf_leave;s--,l+=r[a++]<<u,u+=8}switch(n.last=l&1,l>>>=1,--u,l&3){case 0:n.mode=Y_;break;case 1:if(Tv(n),n.mode=tv,t===D_){l>>>=2,u-=2;break inf_leave}break;case 2:n.mode=Q_;break;case 3:e.msg=`invalid block type`,n.mode=dv}l>>>=2,u-=2;break;case Y_:for(l>>>=u&7,u-=u&7;u<32;){if(s===0)break inf_leave;s--,l+=r[a++]<<u,u+=8}if((l&65535)!=(l>>>16^65535)){e.msg=`invalid stored block lengths`,n.mode=dv;break}if(n.length=l&65535,l=0,u=0,n.mode=X_,t===D_)break inf_leave;case X_:n.mode=Z_;case Z_:if(p=n.length,p){if(p>s&&(p=s),p>c&&(p=c),p===0)break inf_leave;i.set(r.subarray(a,a+p),o),s-=p,a+=p,c-=p,o+=p,n.length-=p;break}n.mode=q_;break;case Q_:for(;u<14;){if(s===0)break inf_leave;s--,l+=r[a++]<<u,u+=8}if(n.nlen=(l&31)+257,l>>>=5,u-=5,n.ndist=(l&31)+1,l>>>=5,u-=5,n.ncode=(l&15)+4,l>>>=4,u-=4,n.nlen>286||n.ndist>30){e.msg=`too many length or distance symbols`,n.mode=dv;break}n.have=0,n.mode=$_;case $_:for(;n.have<n.ncode;){for(;u<3;){if(s===0)break inf_leave;s--,l+=r[a++]<<u,u+=8}n.lens[O[n.have++]]=l&7,l>>>=3,u-=3}for(;n.have<19;)n.lens[O[n.have++]]=0;if(n.lencode=n.lendyn,n.lenbits=7,E={bits:n.lenbits},w=w_(0,n.lens,0,19,n.lencode,0,n.work,E),n.lenbits=E.bits,w){e.msg=`invalid code lengths set`,n.mode=dv;break}n.have=0,n.mode=ev;case ev:for(;n.have<n.nlen+n.ndist;){for(;g=n.lencode[l&(1<<n.lenbits)-1],_=g>>>24,v=g>>>16&255,y=g&65535,!(_<=u);){if(s===0)break inf_leave;s--,l+=r[a++]<<u,u+=8}if(y<16)l>>>=_,u-=_,n.lens[n.have++]=y;else{if(y===16){for(D=_+2;u<D;){if(s===0)break inf_leave;s--,l+=r[a++]<<u,u+=8}if(l>>>=_,u-=_,n.have===0){e.msg=`invalid bit length repeat`,n.mode=dv;break}C=n.lens[n.have-1],p=3+(l&3),l>>>=2,u-=2}else if(y===17){for(D=_+3;u<D;){if(s===0)break inf_leave;s--,l+=r[a++]<<u,u+=8}l>>>=_,u-=_,C=0,p=3+(l&7),l>>>=3,u-=3}else{for(D=_+7;u<D;){if(s===0)break inf_leave;s--,l+=r[a++]<<u,u+=8}l>>>=_,u-=_,C=0,p=11+(l&127),l>>>=7,u-=7}if(n.have+p>n.nlen+n.ndist){e.msg=`invalid bit length repeat`,n.mode=dv;break}for(;p--;)n.lens[n.have++]=C}}if(n.mode===dv)break;if(n.lens[256]===0){e.msg=`invalid code -- missing end-of-block`,n.mode=dv;break}if(n.lenbits=9,E={bits:n.lenbits},w=w_(1,n.lens,0,n.nlen,n.lencode,0,n.work,E),n.lenbits=E.bits,w){e.msg=`invalid literal/lengths set`,n.mode=dv;break}if(n.distbits=6,n.distcode=n.distdyn,E={bits:n.distbits},w=w_(2,n.lens,n.nlen,n.ndist,n.distcode,0,n.work,E),n.distbits=E.bits,w){e.msg=`invalid distances set`,n.mode=dv;break}if(n.mode=tv,t===D_)break inf_leave;case tv:n.mode=nv;case nv:if(s>=6&&c>=258){e.next_out=o,e.avail_out=c,e.next_in=a,e.avail_in=s,n.hold=l,n.bits=u,y_(e,f),o=e.next_out,i=e.output,c=e.avail_out,a=e.next_in,r=e.input,s=e.avail_in,l=n.hold,u=n.bits,n.mode===q_&&(n.back=-1);break}for(n.back=0;g=n.lencode[l&(1<<n.lenbits)-1],_=g>>>24,v=g>>>16&255,y=g&65535,!(_<=u);){if(s===0)break inf_leave;s--,l+=r[a++]<<u,u+=8}if(v&&!(v&240)){for(b=_,x=v,S=y;g=n.lencode[S+((l&(1<<b+x)-1)>>b)],_=g>>>24,v=g>>>16&255,y=g&65535,!(b+_<=u);){if(s===0)break inf_leave;s--,l+=r[a++]<<u,u+=8}l>>>=b,u-=b,n.back+=b}if(l>>>=_,u-=_,n.back+=_,n.length=y,v===0){n.mode=sv;break}if(v&32){n.back=-1,n.mode=q_;break}if(v&64){e.msg=`invalid literal/length code`,n.mode=dv;break}n.extra=v&15,n.mode=rv;case rv:if(n.extra){for(D=n.extra;u<D;){if(s===0)break inf_leave;s--,l+=r[a++]<<u,u+=8}n.length+=l&(1<<n.extra)-1,l>>>=n.extra,u-=n.extra,n.back+=n.extra}n.was=n.length,n.mode=iv;case iv:for(;g=n.distcode[l&(1<<n.distbits)-1],_=g>>>24,v=g>>>16&255,y=g&65535,!(_<=u);){if(s===0)break inf_leave;s--,l+=r[a++]<<u,u+=8}if(!(v&240)){for(b=_,x=v,S=y;g=n.distcode[S+((l&(1<<b+x)-1)>>b)],_=g>>>24,v=g>>>16&255,y=g&65535,!(b+_<=u);){if(s===0)break inf_leave;s--,l+=r[a++]<<u,u+=8}l>>>=b,u-=b,n.back+=b}if(l>>>=_,u-=_,n.back+=_,v&64){e.msg=`invalid distance code`,n.mode=dv;break}n.offset=y,n.extra=v&15,n.mode=av;case av:if(n.extra){for(D=n.extra;u<D;){if(s===0)break inf_leave;s--,l+=r[a++]<<u,u+=8}n.offset+=l&(1<<n.extra)-1,l>>>=n.extra,u-=n.extra,n.back+=n.extra}if(n.offset>n.dmax){e.msg=`invalid distance too far back`,n.mode=dv;break}n.mode=ov;case ov:if(c===0)break inf_leave;if(p=f-c,n.offset>p){if(p=n.offset-p,p>n.whave&&n.sane){e.msg=`invalid distance too far back`,n.mode=dv;break}p>n.wnext?(p-=n.wnext,m=n.wsize-p):m=n.wnext-p,p>n.length&&(p=n.length),h=n.window}else h=i,m=o-n.offset,p=n.length;p>c&&(p=c),c-=p,n.length-=p;do i[o++]=h[m++];while(--p);n.length===0&&(n.mode=nv);break;case sv:if(c===0)break inf_leave;i[o++]=n.length,c--,n.mode=nv;break;case cv:if(n.wrap){for(;u<32;){if(s===0)break inf_leave;s--,l|=r[a++]<<u,u+=8}if(f-=c,e.total_out+=f,n.total+=f,n.wrap&4&&f&&(e.adler=n.check=n.flags?Xh(n.check,i,f,o-f):Jh(n.check,i,f,o-f)),f=c,n.wrap&4&&(n.flags?l:mv(l))!==n.check){e.msg=`incorrect data check`,n.mode=dv;break}l=0,u=0}n.mode=lv;case lv:if(n.wrap&&n.flags){for(;u<32;){if(s===0)break inf_leave;s--,l+=r[a++]<<u,u+=8}if(n.wrap&4&&l!==(n.total&4294967295)){e.msg=`incorrect length check`,n.mode=dv;break}l=0,u=0}n.mode=uv;case uv:w=k_;break inf_leave;case dv:w=M_;break inf_leave;case fv:return N_;case pv:default:return j_}return e.next_out=o,e.avail_out=c,e.next_in=a,e.avail_in=s,n.hold=l,n.bits=u,(n.wsize||f!==e.avail_out&&n.mode<dv&&(n.mode<cv||t!==T_))&&Ev(e,e.output,e.next_out,f-e.avail_out),d-=e.avail_in,f-=e.avail_out,e.total_in+=d,e.total_out+=f,n.total+=f,n.wrap&4&&f&&(e.adler=n.check=n.flags?Xh(n.check,i,f,e.next_out-f):Jh(n.check,i,f,e.next_out-f)),e.data_type=n.bits+(n.last?64:0)+(n.mode===q_?128:0)+(n.mode===tv||n.mode===X_?256:0),(d===0&&f===0||t===T_)&&w===O_&&(w=P_),w},inflateEnd:e=>{if(gv(e))return j_;let t=e.state;return t.window&&=null,e.state=null,O_},inflateGetHeader:(e,t)=>{if(gv(e))return j_;let n=e.state;return n.wrap&2?(n.head=t,t.done=!1,O_):j_},inflateSetDictionary:(e,t)=>{let n=t.length,r,i,a;return gv(e)||(r=e.state,r.wrap!==0&&r.mode!==K_)?j_:r.mode===K_&&(i=1,i=Jh(i,t,n,0),i!==r.check)?M_:(a=Ev(e,t,n,n),a?(r.mode=fv,N_):(r.havedict=1,O_))},inflateInfo:`pako inflate (from Nodeca project)`};function Ov(){this.text=0,this.time=0,this.xflags=0,this.os=0,this.extra=null,this.extra_len=0,this.name=``,this.comment=``,this.hcrc=0,this.done=!1}var kv=Ov;let Av=Object.prototype.toString,{Z_NO_FLUSH:jv,Z_FINISH:Mv,Z_OK:Nv,Z_STREAM_END:Pv,Z_NEED_DICT:Fv,Z_STREAM_ERROR:Iv,Z_DATA_ERROR:Lv,Z_MEM_ERROR:Rv}=Qh;function zv(e){this.options=Yg.assign({chunkSize:65536,windowBits:15,to:``},e||{});let t=this.options;t.raw&&t.windowBits>=0&&t.windowBits<16&&(t.windowBits=-t.windowBits,t.windowBits===0&&(t.windowBits=-15)),t.windowBits>=0&&t.windowBits<16&&!(e&&e.windowBits)&&(t.windowBits+=32),t.windowBits>15&&t.windowBits<48&&(t.windowBits&15||(t.windowBits|=15)),this.err=0,this.msg=``,this.ended=!1,this.chunks=[],this.strm=new n_,this.strm.avail_out=0;let n=Dv.inflateInit2(this.strm,t.windowBits);if(n!==Nv||(this.header=new kv,Dv.inflateGetHeader(this.strm,this.header),t.dictionary&&(typeof t.dictionary==`string`?t.dictionary=e_.string2buf(t.dictionary):Av.call(t.dictionary)===`[object ArrayBuffer]`&&(t.dictionary=new Uint8Array(t.dictionary)),t.raw&&(n=Dv.inflateSetDictionary(this.strm,t.dictionary),n!==Nv))))throw Error(Zh[n])}zv.prototype.push=function(e,t){let n=this.strm,r=this.options.chunkSize,i=this.options.dictionary,a,o,s;if(this.ended)return!1;for(o=t===~~t?t:t===!0?Mv:jv,n.input=Av.call(e)===`[object ArrayBuffer]`?new Uint8Array(e):e,n.next_in=0,n.avail_in=n.input.length;;){for(n.avail_out===0&&(n.output=new Uint8Array(r),n.next_out=0,n.avail_out=r),a=Dv.inflate(n,o),a===Fv&&i&&(a=Dv.inflateSetDictionary(n,i),a===Nv?a=Dv.inflate(n,o):a===Lv&&(a=Fv));n.avail_in>0&&a===Pv&&n.state.wrap>0&&e[n.next_in]!==0;)Dv.inflateReset(n),a=Dv.inflate(n,o);switch(a){case Iv:case Lv:case Fv:case Rv:return this.onEnd(a),this.ended=!0,!1}if(s=n.avail_out,n.next_out&&(n.avail_out===0||a===Pv)){if(this.options.to===`string`){let e=e_.utf8border(n.output,n.next_out),t=n.next_out-e,i=e_.buf2string(n.output,e);n.next_out=t,n.avail_out=r-t,t&&n.output.set(n.output.subarray(e,e+t),0),this.onData(i)}else this.onData(n.output.length===n.next_out?n.output:n.output.subarray(0,n.next_out))}if(a!==Nv||s!==0){if(a===Pv)return a=Dv.inflateEnd(this.strm),this.onEnd(a),this.ended=!0,!0;if(n.avail_in===0)break}}return!0},zv.prototype.onData=function(e){this.chunks.push(e)},zv.prototype.onEnd=function(e){e===Nv&&(this.result=this.options.to===`string`?this.chunks.join(``):Yg.flattenChunks(this.chunks)),this.chunks=[],this.err=e,this.msg=this.strm.msg};function Bv(e,t){let n=new zv(t);if(n.push(e),n.err)throw n.msg||Zh[n.err];return n.result}function Vv(e,t){return t||={},t.raw=!0,Bv(e,t)}var Hv={Inflate:zv,inflate:Bv,inflateRaw:Vv,ungzip:Bv,constants:Qh};let{Deflate:Uv,deflate:Wv,deflateRaw:Gv,gzip:Kv}=__,{Inflate:qv,inflate:Jv,inflateRaw:Yv,ungzip:Xv}=Hv;var Zv=Xv,Qv=class{data;idx=0;decoder=new TextDecoder(`utf-8`);constructor(e){this.data=e}skipWhitespace(){for(;this.idx<this.data.length;){let e=this.data[this.idx];if(e===32||e===10||e===13||e===9){this.idx++;continue}break}}readToken(){this.skipWhitespace();let e=this.idx;for(;this.idx<this.data.length;){let e=this.data[this.idx];if(e===32||e===10||e===13||e===9)break;this.idx++}if(this.idx<=e)throw Error(`Unexpected EOF while reading token`);return this.decoder.decode(this.data.subarray(e,this.idx))}readInt(){let e=this.readToken(),t=Number.parseInt(e,10);if(!Number.isFinite(t))throw Error(`Invalid int token: ${e}`);return t}readFloatAscii(){let e=this.readToken(),t=Number.parseFloat(e);if(!Number.isFinite(t))throw Error(`Invalid float token: ${e}`);return t}readBinaryFloats(e){this.skipWhitespace();let t=this.data.subarray(this.idx,this.idx+5);if(t.length!==5||t[0]!==64||t[1]!==66||t[2]!==73||t[3]!==78||t[4]!==64)throw Error(`Expected @BIN@ marker`);this.idx+=5;let n=e*4,r=this.data.byteOffset+this.idx,i=r+n;if(i>this.data.buffer.byteLength)throw Error(`Unexpected EOF while reading binary floats`);let a=this.data.buffer.slice(r,i);return this.idx+=n,this.skipWhitespace(),new Float32Array(a)}};function $v(e){e.readToken();let t=e.readInt(),n=e.readFloatAscii(),r=e.readInt()!==0,i=e.readInt()!==0,a=e.readBinaryFloats(t),o=e.readBinaryFloats(t),s=r?e.readBinaryFloats(t):new Float32Array(t).fill(1),c=i?e.readBinaryFloats(t):new Float32Array(t).fill(0),l=new Float32Array(t),u=new Float32Array(t);for(let e=0;e<t;e++){let t=s[e]/Math.sqrt(o[e]+n);l[e]=t,u[e]=c[e]-t*a[e]}return{channels:t,mergedScale:l,mergedBias:u}}function ey(e,t){if(e.readToken(),t<11)return`relu`;let n=e.readToken();if(n===`ACTIVATION_IDENTITY`)return`identity`;if(n===`ACTIVATION_RELU`)return`relu`;if(n===`ACTIVATION_MISH`)return`mish`;throw Error(`Unsupported activation kind token: ${n}`)}function ty(e){let t=e.readToken(),n=e.readInt(),r=e.readInt(),i=e.readInt(),a=e.readInt();return{name:t,kernelY:n,kernelX:r,inChannels:i,outChannels:a,dilationY:e.readInt(),dilationX:e.readInt(),weights:e.readBinaryFloats(n*r*i*a)}}function ny(e){let t=e.readToken(),n=e.readInt(),r=e.readInt();return{name:t,inChannels:n,outChannels:r,weights:e.readBinaryFloats(n*r)}}function ry(e){let t=e.readToken(),n=e.readInt();return{name:t,channels:n,weights:e.readBinaryFloats(n)}}function iy(e){let t=new Qv(e),n=t.readToken(),r=t.readInt();if(r<8||r>16)throw Error(`Unsupported modelVersion ${r}, supported 8..16`);let i=t.readInt(),a=t.readInt(),o=r>=13?{tdScoreMultiplier:t.readFloatAscii(),scoreMeanMultiplier:t.readFloatAscii(),scoreStdevMultiplier:t.readFloatAscii(),leadMultiplier:t.readFloatAscii(),varianceTimeMultiplier:t.readFloatAscii(),shorttermValueErrorMultiplier:t.readFloatAscii(),shorttermScoreErrorMultiplier:t.readFloatAscii(),outputScaleMultiplier:1}:{tdScoreMultiplier:20,scoreMeanMultiplier:20,scoreStdevMultiplier:20,leadMultiplier:20,varianceTimeMultiplier:40,shorttermValueErrorMultiplier:.25,shorttermScoreErrorMultiplier:30,outputScaleMultiplier:1},s=r>=15?t.readInt():0;if(r>=15)for(let e=0;e<7;e++)t.readInt();if(s!==0&&s!==1)throw Error(`Unsupported metaEncoderVersion ${s}`);t.readToken();let c=t.readInt(),l=t.readInt(),u=t.readInt(),d=t.readInt();t.readInt();let f=t.readInt();if(r>=15){let e=t.readInt();if(e!==0)throw Error(`Unsupported trunk norm kind ${e}`);for(let e=0;e<5;e++)t.readInt()}let p=ty(t),m=ny(t),h;if(s>0){t.readToken();let e=t.readInt();if(e!==192)throw Error(`Unexpected numInputMetaChannels ${e}`);let n=ny(t),i=ry(t),a=ey(t,r),o=ny(t),s=ry(t),c=ey(t,r),u=ny(t);if(u.outChannels!==l)throw Error(`Meta encoder output ${u.outChannels} does not match trunk channels ${l}`);h={numInputMetaChannels:e,mul1:n,bias1:i,act1:a,mul2:o,bias2:s,act2:c,mul3:u}}function g(){let e=t.readToken();if(e===`ordinary_block`)return t.readToken(),{kind:`ordinary`,preBN:$v(t),preActivation:ey(t,r),w1:ty(t),midBN:$v(t),midActivation:ey(t,r),w2:ty(t)};if(e===`gpool_block`)return t.readToken(),{kind:`gpool`,preBN:$v(t),preActivation:ey(t,r),w1a:ty(t),w1b:ty(t),gpoolBN:$v(t),gpoolActivation:ey(t,r),w1r:ny(t),midBN:$v(t),midActivation:ey(t,r),w2:ty(t)};if(e===`nested_bottleneck_block`){t.readToken();let e=t.readInt(),n=$v(t),i=ey(t,r),a=ty(t),o=[];for(let t=0;t<e;t++)o.push(g());return{kind:`nested_bottleneck`,numBlocks:e,preBN:n,preActivation:i,preConv:a,blocks:o,postBN:$v(t),postActivation:ey(t,r),postConv:ty(t)}}throw Error(`Unsupported trunk block kind ${e}`)}let _=[];for(let e=0;e<c;e++)_.push(g());let v=$v(t),y=ey(t,r);t.readToken();let b=ty(t),x=ty(t),S=$v(t),C=ey(t,r),w=ny(t),T=$v(t),E=ey(t,r),D=ty(t),O=ny(t),k=r>=15?ry(t):void 0,A=r>=15?ey(t,r):void 0,j=r>=15?ny(t):void 0;t.readToken();let M=ty(t),N=$v(t),ee=ey(t,r),P=ny(t),te=ry(t),F=ey(t,r),ne=ny(t),re=ry(t),I=ny(t),ie=ry(t),L=ty(t);return{modelName:n,modelVersion:r,numInputChannels:i,numInputGlobalChannels:a,metaEncoderVersion:s,metaEncoder:h,postProcessParams:o,policyOutChannels:D.outChannels,scoreValueChannels:I.outChannels,trunk:{numBlocks:c,trunkNumChannels:l,midNumChannels:u,regularNumChannels:d,gpoolNumChannels:f,conv1:p,ginput:m,blocks:_,tipBN:v,tipActivation:y},policy:{p1:b,g1:x,g1BN:S,g1Activation:C,gpoolToBias:w,p1BN:T,p1Activation:E,p2:D,passMul:O,passBias:k,passActivation:A,passMul2:j},value:{v1:M,v1BN:N,v1Activation:ee,v2:P,v2Bias:te,v2Activation:F,v3:ne,v3Bias:re,sv3:I,sv3Bias:ie,ownership:L}}}function ay(e){return{scale:Uo(e.mergedScale,[1,1,1,e.channels]),bias:Uo(e.mergedBias,[1,1,1,e.channels])}}function oy(e){let t=Uo(e.weights,[e.kernelY,e.kernelX,e.inChannels,e.outChannels]);return{kernelY:e.kernelY,kernelX:e.kernelX,inChannels:e.inChannels,outChannels:e.outChannels,dilationY:e.dilationY,dilationX:e.dilationX,filter:t}}function sy(e){let t=Ho(e.weights,[e.inChannels,e.outChannels]);return{inChannels:e.inChannels,outChannels:e.outChannels,w:t}}function cy(e){let t=Ho(e.weights,[1,e.channels]);return{channels:e.channels,b:t}}function ly(e,t){return t===`identity`?e:t===`relu`?Vo(e):K(e,Ya(To(e)))}function uy(e,t){return t===`identity`?e:t===`relu`?Vo(e):K(e,Ya(To(e)))}function dy(e,t,n){return ly(G(K(e,t.scale),t.bias),n)}function fy(e,t){return eo(e,t.filter,1,`same`,`NHWC`,[t.dilationY,t.dilationX])}function py(e){let t=((e.shape[1]??19)-14)*.1,n=jo(e,[1,2]),r=ho(e,[1,2]);return Ga([n,K(n,t),r],1)}function my(e){let t=(e.shape[1]??19)-14,n=t*.1,r=t*t*.01-.1,i=jo(e,[1,2]);return Ga([i,K(i,n),K(i,r)],1)}var hy=class{modelName;modelVersion;postProcessParams;policyOutChannels;scoreValueChannels;metaEncoderVersion;metaEncoder;trunkConv1;trunkGInput;trunkBlocks;trunkTipBN;trunkTipActivation;p1;g1;g1BN;g1Activation;gpoolToBias;p1BN;p1Activation;p2;passMul;passBias;passActivation;passMul2;v1;v1BN;v1Activation;v2;v2Bias;v2Activation;v3;v3Bias;sv3;sv3Bias;ownership;constructor(e){this.modelName=e.modelName,this.modelVersion=e.modelVersion,this.postProcessParams=e.postProcessParams,this.policyOutChannels=e.policyOutChannels,this.scoreValueChannels=e.scoreValueChannels,this.metaEncoderVersion=e.metaEncoderVersion,e.metaEncoder&&(this.metaEncoder={mul1:sy(e.metaEncoder.mul1),bias1:cy(e.metaEncoder.bias1),act1:e.metaEncoder.act1,mul2:sy(e.metaEncoder.mul2),bias2:cy(e.metaEncoder.bias2),act2:e.metaEncoder.act2,mul3:sy(e.metaEncoder.mul3)}),this.trunkConv1=oy(e.trunk.conv1),this.trunkGInput=sy(e.trunk.ginput);let t=e=>e.kind===`ordinary`?{kind:`ordinary`,preBN:ay(e.preBN),preActivation:e.preActivation,w1:oy(e.w1),midBN:ay(e.midBN),midActivation:e.midActivation,w2:oy(e.w2)}:e.kind===`gpool`?{kind:`gpool`,preBN:ay(e.preBN),preActivation:e.preActivation,w1a:oy(e.w1a),w1b:oy(e.w1b),gpoolBN:ay(e.gpoolBN),gpoolActivation:e.gpoolActivation,w1r:sy(e.w1r),midBN:ay(e.midBN),midActivation:e.midActivation,w2:oy(e.w2)}:{kind:`nested_bottleneck`,numBlocks:e.numBlocks,preBN:ay(e.preBN),preActivation:e.preActivation,preConv:oy(e.preConv),blocks:e.blocks.map(t),postBN:ay(e.postBN),postActivation:e.postActivation,postConv:oy(e.postConv)};this.trunkBlocks=e.trunk.blocks.map(t),this.trunkTipBN=ay(e.trunk.tipBN),this.trunkTipActivation=e.trunk.tipActivation,this.p1=oy(e.policy.p1),this.g1=oy(e.policy.g1),this.g1BN=ay(e.policy.g1BN),this.g1Activation=e.policy.g1Activation,this.gpoolToBias=sy(e.policy.gpoolToBias),this.p1BN=ay(e.policy.p1BN),this.p1Activation=e.policy.p1Activation,this.p2=oy(e.policy.p2),this.passMul=sy(e.policy.passMul),this.passBias=e.policy.passBias?cy(e.policy.passBias):void 0,this.passActivation=e.policy.passActivation,this.passMul2=e.policy.passMul2?sy(e.policy.passMul2):void 0,this.v1=oy(e.value.v1),this.v1BN=ay(e.value.v1BN),this.v1Activation=e.value.v1Activation,this.v2=sy(e.value.v2),this.v2Bias=cy(e.value.v2Bias),this.v2Activation=e.value.v2Activation,this.v3=sy(e.value.v3),this.v3Bias=cy(e.value.v3Bias),this.sv3=sy(e.value.sv3),this.sv3Bias=cy(e.value.sv3Bias),this.ownership=oy(e.value.ownership)}forward(e,t,n){return Ei(()=>{let r=this.forwardTrunk(e,t,n),i=fy(r,this.p1),a=py(dy(fy(r,this.g1),this.g1BN,this.g1Activation)),o=qa(a,this.gpoolToBias.w);i=G(i,Ua(o,[o.shape[0],1,1,o.shape[1]]));let s=fy(dy(i,this.p1BN,this.p1Activation),this.p2),c=this.forwardPolicyPass(a),l=dy(fy(r,this.v1),this.v1BN,this.v1Activation),u=my(l),d=qa(u,this.v2.w);d=G(d,this.v2Bias.b),d=uy(d,this.v2Activation);let f=qa(d,this.v3.w);f=G(f,this.v3Bias.b);let p=qa(d,this.sv3.w);p=G(p,this.sv3Bias.b);let m=fy(l,this.ownership);return{policy:s,policyPass:c,value:f,scoreValue:p,ownership:m}})}forwardPolicyValue(e,t,n){return Ei(()=>{let r=this.forwardTrunk(e,t,n),i=fy(r,this.p1),a=py(dy(fy(r,this.g1),this.g1BN,this.g1Activation)),o=qa(a,this.gpoolToBias.w);i=G(i,Ua(o,[o.shape[0],1,1,o.shape[1]]));let s=fy(dy(i,this.p1BN,this.p1Activation),this.p2),c=this.forwardPolicyPass(a),l=my(dy(fy(r,this.v1),this.v1BN,this.v1Activation)),u=qa(l,this.v2.w);u=G(u,this.v2Bias.b),u=uy(u,this.v2Activation);let d=qa(u,this.v3.w);d=G(d,this.v3Bias.b);let f=qa(u,this.sv3.w);return f=G(f,this.sv3Bias.b),{policy:s,policyPass:c,value:d,scoreValue:f}})}forwardValueOnly(e,t,n){return Ei(()=>{let r=my(dy(fy(this.forwardTrunk(e,t,n),this.v1),this.v1BN,this.v1Activation)),i=qa(r,this.v2.w);i=G(i,this.v2Bias.b),i=uy(i,this.v2Activation);let a=qa(i,this.v3.w);a=G(a,this.v3Bias.b);let o=qa(i,this.sv3.w);return o=G(o,this.sv3Bias.b),{value:a,scoreValue:o}})}forwardTrunk(e,t,n){let r=fy(e,this.trunkConv1),i=qa(t,this.trunkGInput.w);if(r=G(r,Ua(i,[i.shape[0],1,1,i.shape[1]])),this.metaEncoder){if(!n)throw Error(`This model needs SGF metadata input`);let e=this.forwardMetaEncoder(n);r=G(r,Ua(e,[e.shape[0],1,1,e.shape[1]]))}return r=this.applyBlockStack(r,this.trunkBlocks),dy(r,this.trunkTipBN,this.trunkTipActivation)}forwardMetaEncoder(e){let t=this.metaEncoder,n=qa(e,t.mul1.w);return n=G(n,t.bias1.b),n=uy(n,t.act1),n=qa(n,t.mul2.w),n=G(n,t.bias2.b),n=uy(n,t.act2),qa(n,t.mul3.w)}forwardPolicyPass(e){let t=qa(e,this.passMul.w);return this.passBias&&this.passActivation&&this.passMul2&&(t=G(t,this.passBias.b),t=uy(t,this.passActivation),t=qa(t,this.passMul2.w)),t}applyBlockStack(e,t){for(let n of t){if(n.kind===`ordinary`){let t=fy(dy(fy(dy(e,n.preBN,n.preActivation),n.w1),n.midBN,n.midActivation),n.w2);e=G(e,t);continue}if(n.kind===`gpool`){let t=dy(e,n.preBN,n.preActivation),r=fy(t,n.w1a),i=py(dy(fy(t,n.w1b),n.gpoolBN,n.gpoolActivation)),a=qa(i,n.w1r.w);r=G(r,Ua(a,[a.shape[0],1,1,a.shape[1]]));let o=fy(dy(r,n.midBN,n.midActivation),n.w2);e=G(e,o);continue}let t=fy(dy(e,n.preBN,n.preActivation),n.preConv);t=this.applyBlockStack(t,n.blocks);let r=fy(dy(t,n.postBN,n.postActivation),n.postConv);e=G(e,r)}return e}dispose(){let e=[this.trunkConv1.filter,this.trunkGInput.w,this.trunkTipBN.scale,this.trunkTipBN.bias,this.p1.filter,this.g1.filter,this.g1BN.scale,this.g1BN.bias,this.gpoolToBias.w,this.p1BN.scale,this.p1BN.bias,this.p2.filter,this.passMul.w,...this.passBias?[this.passBias.b]:[],...this.passMul2?[this.passMul2.w]:[],this.v1.filter,this.v1BN.scale,this.v1BN.bias,this.v2.w,this.v2Bias.b,this.v3.w,this.v3Bias.b,this.sv3.w,this.sv3Bias.b,this.ownership.filter],t=n=>{if(e.push(n.preBN.scale,n.preBN.bias),n.kind===`ordinary`){e.push(n.w1.filter,n.midBN.scale,n.midBN.bias,n.w2.filter);return}if(n.kind===`gpool`){e.push(n.w1a.filter,n.w1b.filter,n.gpoolBN.scale,n.gpoolBN.bias,n.w1r.w,n.midBN.scale,n.midBN.bias,n.w2.filter);return}e.push(n.preConv.filter);for(let e of n.blocks)t(e);e.push(n.postBN.scale,n.postBN.bias,n.postConv.filter)};for(let e of this.trunkBlocks)t(e);Di(e)}};let gy=e=>e>20?e:e<-20?Math.exp(e):Math.log1p(Math.exp(e));function _y(e){let{nextPlayer:t,valueLogits:n,scoreValue:r}=e,i=e.postProcessParams,a=i?.outputScaleMultiplier??1,o=n[0]*a,s=n[1]*a,c=n[2]*a,l=Math.max(o,s,c),u=Math.exp(o-l),d=Math.exp(s-l),f=Math.exp(c-l),p=u+d+f;u/=p,d/=p,f/=p;let m=i?.scoreMeanMultiplier??20,h=i?.scoreStdevMultiplier??20,g=i?.leadMultiplier??20,_=r[0]*a,v=r[1]*a,y=r[2]*a,b=_*m,x=gy(v)*h,S=b*b+x*x,C=y*g,w=i?.varianceTimeMultiplier??40,T=(r[3]??0)*a,E=r.length>=4?gy(T)*w:-1;b*=1-f,S*=1-f,C*=1-f;let D=t===`black`?u:d,O=t===`black`?C:-C,k=t===`black`?b:-b,A=Math.sqrt(Math.max(0,S-b*b)),j=f,M=-1,N=-1;if(r.length>=6){let t=e.modelVersion??0,n=i?.shorttermValueErrorMultiplier??.25,o=i?.shorttermScoreErrorMultiplier??30,s=r[4]*a,c=r[5]*a;if(t>=14){let e=gy(s*.5);M=Math.sqrt(e*e*n);let t=gy(c*.5);N=Math.sqrt(t*t*o)}else t>=10?(M=Math.sqrt(gy(s)*n),N=Math.sqrt(gy(c)*o)):(M=gy(s),N=gy(c)*10)}return{blackWinProb:D,blackScoreLead:O,blackScoreMean:k,blackScoreStdev:A,blackNoResultProb:j,varTimeLeft:E,shorttermWinlossError:M,shorttermScoreError:N}}async function vy(e){let t=[...new Uint8Array(await crypto.subtle.digest(`SHA-256`,e))].map(e=>e.toString(16).padStart(2,`0`)).join(``),n=t===`f5d32604e3675c480c7c8f6aa579a1ea857135628a0afccc8fa56330fbacd38d`?Zv(e):t===`2bcc7c4daac6e33e9b9f5f8a427e1536b50aa705cfdb7fe21bffacbcdbfc7242`?e:null;if(!n)throw Error(`Model SHA-256 mismatch`);return new hy(iy(n))}async function yy(e,t){let n=performance.now(),r=Uo(t.spatial,[1,t.size,t.size,22]),i=Ho(t.global,[1,19]),a;try{a=e.forward(r,i);let[o,s,c,l,u]=await Promise.all([a.policy.data(),a.policyPass.data(),a.value.data(),a.scoreValue.data(),a.ownership.data()]),d=Array.from({length:t.size**2},(t,n)=>o[n*e.policyOutChannels]);d.push(s[0]);let f=_y({nextPlayer:t.nextPlayer,valueLogits:c,scoreValue:l,postProcessParams:e.postProcessParams});if([...d,...u,...Object.values(f)].some(e=>!Number.isFinite(e)))throw Error(`Non-finite network output`);return{logits:d,ownership:Array.from(u,e=>Math.tanh(e)),evaluation:f,inferenceMs:performance.now()-n}}finally{r.dispose(),i.dispose(),a&&Object.values(a).forEach(e=>e.dispose())}}
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
var X;(function(e){e[e.float32=0]=`float32`,e[e.int32=1]=`int32`,e[e.bool=2]=`bool`,e[e.string=3]=`string`,e[e.complex64=4]=`complex64`})(X||={});var by;(function(e){e[e.linear=0]=`linear`,e[e.relu=1]=`relu`,e[e.relu6=2]=`relu6`,e[e.prelu=3]=`prelu`,e[e.leakyrelu=4]=`leakyrelu`,e[e.sigmoid=5]=`sigmoid`,e[e.elu=6]=`elu`})(by||={});
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let xy;function Sy(e){xy=e.wasm.cwrap(Kn,null,[`number`,`array`,`number`,`number`,`array`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`])}function Cy(e){let{inputs:t,backend:n,attrs:r}=e,{a:i,b:a,bias:o,preluActivationWeights:s}=t;if(i.dtype!==`float32`||a.dtype!==`float32`)throw Error(`_FusedMatMul for non non-float32 tensors not yet supported.`);let{transposeA:c,transposeB:l,activation:u,leakyreluAlpha:d}=r,f=n.dataIdMap.get(i.dataId).id,p=n.dataIdMap.get(a.dataId).id,m=0;if(o!=null){let e=n.dataIdMap.get(o.dataId);if(e.shape.length!==1)throw Error(`_FusedMatMul only supports rank-1 bias but got rank ${e.shape.length}.`);m=e.id}let h=s==null?0:n.dataIdMap.get(s.dataId).id,g=by[u];if(g==null)throw Error(`${u} activation not yet supported for FusedConv2D in the wasm backend.`);let _=c?i.shape[2]:i.shape[1],v=l?a.shape[1]:a.shape[2],y=no(i.shape.slice(0,-2),a.shape.slice(0,-2)),b=n.makeOutput([...y,_,v],i.dtype),x=n.dataIdMap.get(b.dataId).id,S=new Uint8Array(new Int32Array(i.shape).buffer),C=new Uint8Array(new Int32Array(a.shape).buffer);return xy(f,S,i.shape.length,p,C,a.shape.length,c,l,g,m,h,d||0,x),b}let wy={kernelName:Kn,backendName:`wasm`,setupFunc:Sy,kernelFunc:Cy};
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Z(e,t){let n;function r(t){n=t.wasm.cwrap(e,null,[`number`,`number`,`number`])}function i(e){let{backend:r,inputs:{x:i}}=e,a=r.dataIdMap.get(i.dataId).id,o=r.makeOutput(i.shape,t||i.dtype),s=r.dataIdMap.get(o.dataId).id;return _(o.shape)===0||n(a,X[i.dtype],s),o}return{kernelName:e,backendName:`wasm`,setupFunc:r,kernelFunc:i}}
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let Ty=Z(`Abs`),Ey=Z(ve),Dy=Z(ye)
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Oy(e,t,n){let r;function i(t){r=t.wasm.cwrap(e,null,[`number`,`array`,`number`,`number`,`array`,`number`,`number`,`number`])}function a(e){let{backend:t,inputs:i}=e,{a,b:o}=i,s=t.dataIdMap.get(a.dataId).id,c=t.dataIdMap.get(o.dataId).id,l=n??a.dtype,u=no(a.shape,o.shape),d=t.makeOutput(u,l);if(_(u)===0)return d;let f=new Uint8Array(new Int32Array(a.shape).buffer),p=new Uint8Array(new Int32Array(o.shape).buffer),m=t.dataIdMap.get(d.dataId).id;return r(s,f,a.shape.length,c,p,o.shape.length,X[a.dtype],m),d}return{kernelName:e,backendName:`wasm`,setupFunc:i,kernelFunc:a}}
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let ky=Oy(`Add`,!0),Ay
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;function jy(e){Ay=e.wasm.cwrap(be,null,[`array`,`number`,`number`,`number`])}function My(e){let{inputs:t,backend:n}=e,r=n.makeOutput(t[0].shape,t[0].dtype);if(_(r.shape)===0)return r;let i=t.map(e=>n.dataIdMap.get(e.dataId).id),a=new Uint8Array(new Int32Array(i).buffer),o=n.dataIdMap.get(r.dataId).id;return Ay(a,i.length,X[r.dtype],o),r}let Ny={kernelName:be,backendName:`wasm`,setupFunc:jy,kernelFunc:My};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Py(e){let{inputs:{x:t},backend:n}=e;if(t.dtype===`string`)return Si(n.readSync(t.dataId),t.shape,t.dtype);let r=n.makeOutput(t.shape,t.dtype),i=n.typedArrayFromHeap(t);return n.typedArrayFromHeap(r).set(i),r}let Fy={kernelName:_t,backendName:`wasm`,kernelFunc:Py},Iy
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;function Ly(e){Iy=e.wasm.cwrap(Bn,null,[`number`,`array`,`number`,`number`,`number`,`array`,`number`])}function Ry(e){let{inputs:t,backend:n,attrs:r}=e,[i,a]=By(t.x.shape,r.perm),o=!0;for(let e=0;e<a.length;e++)a[e]!==e&&(o=!1);let s=zy(t.x.shape,r.perm),c={dataId:t.x.dataId,shape:i,dtype:t.x.dtype};if(o){let e=Py({inputs:t,backend:n});return e.shape=s,e}let l=n.makeOutput(s,c.dtype),u=n.dataIdMap.get(c.dataId).id,d=n.dataIdMap.get(l.dataId).id,f=new Uint8Array(new Int32Array(a).buffer),p=new Uint8Array(new Int32Array(c.shape).buffer);return Iy(u,p,c.shape.length,X[c.dtype],d,f,a.length),l}function zy(e,t){let n=Array(e.length);for(let r=0;r<n.length;r++)n[r]=e[t[r]];return n}function By(e,t){let n=[],r=[];for(let i=0;i<e.length;++i)e[i]!==1&&n.push(e[i]),e[t[i]]!==1&&r.push(t[i]);for(let e=0;e<r.length;++e){let t=-1;for(let n=0;n<r.length;++n)r[n]>=e&&(t===-1||r[t]>r[n])&&(t=n);r[t]=e}return[n,r]}let Vy={kernelName:Bn,backendName:`wasm`,kernelFunc:Ry,setupFunc:Ly};
/**
* @license
* Copyright 2020 Google Inc. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Hy(e,t,n){let r=e.shape,i=e.shape.length,a=S(t,r),o=a,s=uo(o,i),c=null,l=!1;if(s!=null){let t=Array(i);for(let e=0;e<t.length;e++)t[e]=r[s[e]];o=po(o.length,i),c=Ry({inputs:{x:e},attrs:{perm:s},backend:n});let a=n.dataIdMap.get(e.dataId).id;n.dataIdMap.get(c.dataId).id!==a&&(l=!0)}return{transposed:c,originalAxes:a,axes:o,inputWasTransposed:l}}
/**
* @license
* Copyright 2021 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let Uy;function Wy(e){Uy=e.wasm.cwrap(`All`,null,[`number, number, number`])}function Gy(e){let{backend:t,inputs:n,attrs:r}=e,{axis:i,keepDims:a}=r,{x:o}=n,s=t.dataIdMap.get(o.dataId).id,c=o,{transposed:l,axes:u,originalAxes:d,inputWasTransposed:f}=Hy(o,i,t);if(f){let e=t.dataIdMap.get(l.dataId).id;c=l,s=e}let p=c.shape.length;lo(`all`,u,p);let[m,h]=so(c.shape,u),g=_(h),v=t.makeOutput(m,o.dtype);if(_(c.shape)!==0){let e=t.dataIdMap.get(v.dataId).id;Uy(s,g,e)}return f&&t.disposeData(l.dataId),a&&(v.shape=co(v.shape,d)),v}let Ky={kernelName:`All`,backendName:`wasm`,setupFunc:Wy,kernelFunc:Gy},qy
/**
* @license
* Copyright 2021 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;function Jy(e){qy=e.wasm.cwrap(`Any`,null,[`number, number, number`])}function Yy(e){let{backend:t,inputs:n,attrs:r}=e,{axis:i,keepDims:a}=r,{x:o}=n,s=t.dataIdMap.get(o.dataId).id,c=o,{transposed:l,axes:u,originalAxes:d,inputWasTransposed:f}=Hy(o,i,t);if(f){let e=t.dataIdMap.get(l.dataId).id;c=l,s=e}let p=c.shape.length;lo(`any`,u,p);let[m,h]=so(c.shape,u),g=_(h),v=t.makeOutput(m,o.dtype);if(_(c.shape)!==0){let e=t.dataIdMap.get(v.dataId).id;qy(s,g,e)}return f&&t.disposeData(l.dataId),a&&(v.shape=co(v.shape,d)),v}let Xy={kernelName:`Any`,backendName:`wasm`,setupFunc:Jy,kernelFunc:Yy};
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Zy(e){let t;function n(n){t=n.wasm.cwrap(e,null,[`number`,`number`,`number`,`number`,`number`])}function r(e){let{backend:n,inputs:r,attrs:i}=e,{axis:a}=i,{x:o}=r,s=n.dataIdMap.get(o.dataId).id,c=s,l=o,{transposed:u,axes:d,inputWasTransposed:f}=Hy(o,a,n);if(f){let e=n.dataIdMap.get(u.dataId).id;e!==s&&(l=u,c=e)}let p=l.shape.slice(0,-1),m=n.makeOutput(p,`int32`),h=n.dataIdMap.get(m.dataId).id,g=_(m.shape),v=l.shape[d[0]];return t(c,X[l.dtype],g,v,h),f&&n.disposeData(u.dataId),m}return{kernelName:e,backendName:`wasm`,setupFunc:n,kernelFunc:r}}
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let Qy=Zy(xe),$y=Zy(Se),eb=Z(Ce),tb=Z(we),nb=Z(Te),rb=Oy(De,!1),ib=Z(Ee),ab
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;function ob(e){ab=e.wasm.cwrap(Oe,null,[`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`])}function sb(e){let{inputs:t,attrs:n,backend:r}=e,i=t.x,a=r.dataIdMap.get(i.dataId).id,{filterSize:o,strides:s,pad:c,dimRoundingMode:l}=n,u=wa(i.shape,o,s,1,c,l),d=u.filterHeight,f=u.filterWidth,p=u.padInfo.top,m=u.padInfo.right,h=u.padInfo.bottom,g=u.padInfo.left,_=u.strideHeight,v=u.strideWidth,y=u.inChannels;if(u.dataFormat!==`channelsLast`)throw Error(`wasm backend does not support dataFormat:'${u.dataFormat}'. Please use 'channelsLast'.`);if(u.dilationWidth!==1||u.dilationHeight!==1)throw Error(`was backend only supports average pooling with dilation = [1, 1], got [${u.dilationHeight}, ${u.dilationWidth}].`);let b=r.makeOutput(u.outShape,`float32`),x=r.dataIdMap.get(b.dataId).id;return ab(a,i.shape[0],i.shape[1],i.shape[2],d,f,p,m,h,g,_,v,y,x),b}let cb={kernelName:Oe,backendName:`wasm`,setupFunc:ob,kernelFunc:sb},lb
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;function ub(e){lb=e.wasm.cwrap(`AvgPool3D`,null,[`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`])}function db(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{filterSize:a,strides:o,pad:s,dimRoundingMode:c,dataFormat:l}=r,u=Ta(i.shape,a,o,1,s,c,l),d=n.makeOutput(u.outShape,i.dtype);return lb(n.dataIdMap.get(i.dataId).id,n.dataIdMap.get(d.dataId).id,u.batchSize,u.inChannels,u.inDepth,u.inHeight,u.inWidth,u.outDepth,u.outHeight,u.outWidth,u.strideDepth,u.strideHeight,u.strideWidth,u.dilationDepth,u.dilationHeight,u.dilationWidth,u.effectiveFilterDepth,u.effectiveFilterHeight,u.effectiveFilterWidth,u.padInfo.front,u.padInfo.top,u.padInfo.left),d}let fb={kernelName:Ae,backendName:`wasm`,setupFunc:ub,kernelFunc:db},pb
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;function mb(e){pb=e.wasm.cwrap(`AvgPool3DGrad`,null,[`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`])}function hb(e){let{inputs:t,backend:n,attrs:r}=e,{dy:i,input:a}=t,{filterSize:o,strides:s,pad:c,dimRoundingMode:l}=r,u=Ta(a.shape,o,s,1,c,l),d=n.makeOutput(a.shape,a.dtype);return pb(n.dataIdMap.get(i.dataId).id,n.dataIdMap.get(d.dataId).id,u.batchSize,u.inChannels,u.inDepth,u.inHeight,u.inWidth,u.outDepth,u.outHeight,u.outWidth,u.strideDepth,u.strideHeight,u.strideWidth,u.dilationDepth,u.dilationHeight,u.dilationWidth,u.effectiveFilterDepth,u.effectiveFilterHeight,u.effectiveFilterWidth,u.padInfo.front,u.padInfo.top,u.padInfo.left,u.filterDepth,u.filterHeight,u.filterWidth),d}let gb={kernelName:je,backendName:`wasm`,setupFunc:mb,kernelFunc:hb},_b
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;function vb(e){_b=e.wasm.cwrap(`AvgPoolGrad`,null,[`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`])}function yb(e){let{inputs:t,backend:n,attrs:r}=e,{dy:i,input:a}=t,{filterSize:o,strides:s,pad:c}=r,l=wa(a.shape,o,s,1,c),u=n.makeOutput(a.shape,a.dtype);return _b(n.dataIdMap.get(i.dataId).id,n.dataIdMap.get(u.dataId).id,l.batchSize,l.inChannels,l.inHeight,l.inWidth,l.outHeight,l.outWidth,l.strideHeight,l.strideWidth,l.dilationHeight,l.dilationWidth,l.effectiveFilterHeight,l.effectiveFilterWidth,l.padInfo.top,l.padInfo.left,l.filterHeight,l.filterWidth),u}let bb={kernelName:ke,backendName:`wasm`,setupFunc:vb,kernelFunc:yb};
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function xb(e){let{inputs:t,attrs:n}=e,{x:r}=t,{shape:i}=n,a=_(r.shape),o=x(i,a);return m(a===_(o),()=>`new shape: ${o}, old shape: ${r.shape}. New shape and old shape must have the same number of elements.`),e.backend.incRef(r.dataId),{dataId:r.dataId,shape:o,dtype:r.dtype}}let Sb={kernelName:en,backendName:`wasm`,kernelFunc:xb},Cb
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;function wb(e){Cb=e.wasm.cwrap(Me,null,[`number`,`array`,`number`,`number`,`array`,`number`,`number`,`number`,`number`])}function Tb(e){let{inputs:t,backend:n,attrs:r}=e,{a:i,b:a}=t,{transposeA:o,transposeB:s}=r;if(i.dtype!==`float32`||a.dtype!==`float32`)throw Error(`BatchMatMul for non non-float32 tensors not yet supported.`);let c=i.shape.length,l=a.shape.length,u=o?i.shape[c-2]:i.shape[c-1],d=s?a.shape[l-1]:a.shape[l-2],f=o?i.shape[c-1]:i.shape[c-2],p=s?a.shape[l-2]:a.shape[l-1],h=i.shape.slice(0,-2),g=a.shape.slice(0,-2),v=_(h),y=_(g),b=no(i.shape.slice(0,-2),a.shape.slice(0,-2)).concat([f,p]);m(u===d,()=>`Error in matMul: inner shapes (${u}) and (${d}) of Tensors with shapes ${i.shape} and ${a.shape} and transposeA=${o} and transposeB=${s} must match.`);let x=o?[v,u,f]:[v,f,u],S=s?[y,p,d]:[y,d,p],C=xb({inputs:{x:i},backend:n,attrs:{shape:x}}),w=xb({inputs:{x:a},backend:n,attrs:{shape:S}}),T=n.dataIdMap.get(C.dataId).id,E=n.dataIdMap.get(w.dataId).id,D=o?C.shape[2]:C.shape[1],O=s?w.shape[1]:w.shape[2],k=Math.max(v,y),A=n.makeOutput([k,D,O],C.dtype),j=n.dataIdMap.get(A.dataId).id,M=new Uint8Array(new Int32Array(C.shape).buffer),N=new Uint8Array(new Int32Array(w.shape).buffer);return Cb(T,M,C.shape.length,E,N,w.shape.length,o,s,j),n.disposeData(C.dataId),n.disposeData(w.dataId),A.shape=b,A}let Eb={kernelName:Me,backendName:`wasm`,setupFunc:wb,kernelFunc:Tb};
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Db(e){let{inputs:{x:t},attrs:{begin:n,size:r},backend:i}=e,[a,o]=xs(t,n,r),s=ys(t.shape,a,o),c=i.readSync(t.dataId),l=i.makeOutput(o,t.dtype),u=P(t.shape),d=i.dataIdMap.get(l.dataId);if(s){let e=bs(a,u);return t.dtype===`string`?d.stringBytes=c.slice(e,e+_(o)):i.typedArrayFromHeap(l).set(c.subarray(e,e+_(o))),l}if(t.dtype===`string`)return d.stringBytes=zl(c,a,o,t.shape,t.dtype),l;let f=i.typedArrayFromHeap(l),p=t.shape.length;if(p===2)Ob(c,u[0],f,a,o);else if(p===3)kb(c,u[0],u[1],f,a,o);else if(p===4)Ab(c,u[0],u[1],u[2],f,a,o);else{let e=zl(c,a,o,t.shape,t.dtype);f.set(e)}return l}function Ob(e,t,n,r,i){let a=0,o=r[0],s=r[1],c=o+i[0];for(let r=o;r<c;r++){let o=r*t+s;n.set(e.subarray(o,o+i[1]),a),a+=i[1]}}function kb(e,t,n,r,i,a){let o=0,s=i[0],c=i[1],l=i[2],u=s+a[0],d=c+a[1];for(let i=s;i<u;i++)for(let s=c;s<d;s++){let c=i*t+s*n+l;r.set(e.subarray(c,c+a[2]),o),o+=a[2]}}function Ab(e,t,n,r,i,a,o){let s=0,c=a[0],l=a[1],u=a[2],d=c+o[0],f=l+o[1],p=u+o[2],m=a[3];for(let a=c;a<d;a++)for(let c=l;c<f;c++)for(let l=u;l<p;l++){let u=a*t+c*n+l*r+m;i.set(e.subarray(u,u+o[3]),s),s+=o[3]}}let jb={kernelName:hn,backendName:`wasm`,kernelFunc:Db};
/**
* @license
* Copyright 2021 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Mb(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{blockShape:a,crops:o}=r,s=a.reduce((e,t)=>e*t),c=Ns(i.shape,a,s),l=Ps(c.length,a.length),u=Fs(i.shape,a,s),d=Is(o,a.length),f=Ls(u,o,a.length),p=xb({inputs:{x:i},backend:n,attrs:{shape:c}}),m=Ry({inputs:{x:p},backend:n,attrs:{perm:l}}),h=xb({inputs:{x:m},backend:n,attrs:{shape:u}}),g=Db({inputs:{x:h},backend:n,attrs:{begin:d,size:f}});return n.disposeData(p.dataId),n.disposeData(m.dataId),n.disposeData(h.dataId),g}let Nb={kernelName:Ne,backendName:`wasm`,kernelFunc:Mb},Pb
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;function Fb(e){Pb=e.wasm.cwrap(Pe,null,[`number`,`number`,`boolean`,`number`,`number`,`number`])}function Ib(e){let{backend:t,inputs:n,attrs:r}=e,{x:i,weights:a}=n,{size:o}=r,s=a.shape.reduce((e,t)=>e*t,1)!==0,c=i.shape.length===1?[o]:[i.shape[0],o],l=t.makeOutput(c,a.dtype);function u(e){return t.dataIdMap.get(e.dataId).id}return Pb(u(i),o,s,u(a),X[a.dtype],u(l)),l}let Lb={kernelName:Pe,backendName:`wasm`,setupFunc:Fb,kernelFunc:Ib},Rb=Oy(Fe,!0)
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function zb(e){let{inputs:t,backend:n}=e,{s0:r,s1:i}=t,a=n.typedArrayFromHeap(r),o=n.typedArrayFromHeap(i),s=no(Array.from(a),Array.from(o));return n.makeOutput([s.length],`int32`,void 0,new Int32Array(s))}let Bb={kernelName:Ie,backendName:`wasm`,kernelFunc:zb};
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Vb(e){let{inputs:{x:t},attrs:{dtype:n},backend:r}=e,i=r.makeOutput(t.shape,n),a=r.typedArrayFromHeap(t);return r.typedArrayFromHeap(i).set(a),i}let Hb={kernelName:Le,backendName:`wasm`,kernelFunc:Vb},Ub=Z(Re),Wb
/**
* @license
* Copyright 2021 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;function Gb(e){Wb=e.wasm.cwrap(ze,null,[`number`,`number`,`number`,`number`])}function Kb(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{clipValueMin:a,clipValueMax:o}=r,s=n.dataIdMap.get(i.dataId).id,c=n.makeOutput(i.shape,i.dtype),l=n.dataIdMap.get(c.dataId).id;return Wb(s,a,o,l),c}let qb={kernelName:ze,backendName:`wasm`,setupFunc:Gb,kernelFunc:Kb};
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Jb(e){let{inputs:t,backend:n}=e,r=S(e.attrs.axis,t[0].shape)[0];Ts(t.map(e=>e.shape),r);let i=Es(t.map(e=>e.shape),r),a=t.filter(e=>_(e.shape)>0);if(a.length===1)return Py({inputs:{x:a[0]},backend:n});let o=n.makeOutput(i,t[0].dtype);if(_(i)===0)return o;if(a[0].dtype===`string`){let e=a.map(e=>{let t=[-1,_(e.shape.slice(r))];return xb({inputs:{x:e},backend:n,attrs:{shape:t}})}),s=e.map(e=>({vals:n.readSync(e.dataId),shape:e.shape}));i=Es(e.map(e=>e.shape),1);let c=e[0].shape[0]===1,l=Bc(s,i,t[0].dtype,c);o.shape=Es(a.map(e=>e.shape),r);let u=n.dataIdMap.get(o.dataId);return u.stringBytes=mc(l),e.forEach(e=>n.disposeData(e.dataId)),o}let s=_(a[0].shape.slice(0,r)),c=0,l=a.map(e=>{let t=_(e.shape.slice(r));return c+=t,t}),u=a.map(e=>n.typedArrayFromHeap(e)),d=n.typedArrayFromHeap(o);for(let e=0;e<s;e++){let t=e*c;for(let n=0;n<u.length;n++){let r=l[n],i=e*r,a=u[n].subarray(i,i+r);d.set(a,t),t+=r}}return o}let Yb={kernelName:Ve,backendName:`wasm`,kernelFunc:Jb},Xb
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;function Zb(e){Xb=e.wasm.cwrap(He,null,[`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`])}function Qb(e){let{inputs:t,attrs:n,backend:r}=e,{x:i,filter:a}=t,o=r.dataIdMap.get(i.dataId).id,s=r.dataIdMap.get(a.dataId).id,{strides:c,dilations:l,pad:u,dimRoundingMode:d,dataFormat:f}=n,p=Ba(f),m=Ea(i.shape,a.shape,c,l,u,d,!1,p),h=m.filterHeight,g=m.filterWidth,_=m.padInfo.top,v=m.padInfo.right,y=m.padInfo.bottom,b=m.padInfo.left,x=m.dilationHeight,S=m.dilationWidth,C=m.strideHeight,w=m.strideWidth,T=m.inChannels,E=m.outChannels,D=+(m.padInfo.type===`SAME`);if(m.dataFormat!==`channelsLast`)throw Error(`wasm backend Conv2D does not support dataFormat:'${m.dataFormat}'. Please use 'channelsLast'.`);let O=r.makeOutput(m.outShape,`float32`),k=r.dataIdMap.get(O.dataId).id;return Xb(o,i.shape[0],i.shape[1],i.shape[2],s,h,g,_,v,y,b,D,x,S,C,w,T,E,k),O}let $b={kernelName:He,backendName:`wasm`,setupFunc:Zb,kernelFunc:Qb},ex
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;function tx(e){ex=e.wasm.cwrap(Ue,null,`number.number.number.number.number.number.number.number.number.number.number.number.number.number.number.number.number.number.number.number.number.number.number.number.number.number.number`.split(`.`))}function nx(e){let{backend:t,inputs:n,attrs:r}=e,{dy:i,filter:a}=n,{strides:o,pad:s,dataFormat:c,dimRoundingMode:l,inputShape:u}=r,d=Ba(c),f=Ea(u,a.shape,o,1,s,l,!1,d),{batchSize:p,filterHeight:m,filterWidth:h,inChannels:g,inHeight:_,inWidth:v,outChannels:y,outHeight:b,outWidth:x,strideHeight:S,strideWidth:C}=f,w=m-1-f.padInfo.top,T=h-1-f.padInfo.left,E=f.dataFormat===`channelsLast`,D=P(f.inShape),O=P(i.shape),[k,A,j]=P(a.shape),M=D[0],N=E?D[1]:D[2],ee=E?D[2]:1,te=E?1:D[1],F=O[0],ne=E?O[1]:O[2],re=E?O[2]:1,I=E?1:O[1],ie=t.makeOutput(f.inShape,`float32`),L=t.dataIdMap.get(ie.dataId).id,R=t.dataIdMap.get(i.dataId).id,ae=t.dataIdMap.get(a.dataId).id;return ex(R,ae,p,m,h,_,v,g,b,x,y,S,C,w,T,k,A,j,M,N,ee,te,F,ne,re,I,L),ie}let rx={kernelName:Ue,backendName:`wasm`,setupFunc:tx,kernelFunc:nx},ix
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;function ax(e){ix=e.wasm.cwrap(We,null,[`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`])}function ox(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,filter:a}=t,{strides:o,pad:s,dilations:c}=r;if(i.dtype!==`float32`)throw Error(`Tensor x must have dtype float32, got ${i.dtype}`);if(a.dtype!==`float32`)throw Error(`Tensor filter must have dtype float32, got ${a.dtype}`);let l=Da(i.shape,a.shape,o,c,s),u=n.makeOutput(l.outShape,i.dtype);return ix(n.dataIdMap.get(i.dataId).id,n.dataIdMap.get(a.dataId).id,n.dataIdMap.get(u.dataId).id,l.batchSize,l.inDepth,l.inHeight,l.inWidth,l.inChannels,l.outDepth,l.outHeight,l.outWidth,l.outChannels,l.strideDepth,l.strideHeight,l.strideWidth,l.dilationDepth,l.dilationHeight,l.dilationWidth,l.filterDepth,l.filterHeight,l.filterWidth,l.padInfo.front,l.padInfo.top,l.padInfo.left),u}let sx={kernelName:We,backendName:`wasm`,setupFunc:ax,kernelFunc:ox},cx
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;function lx(e){cx=e.wasm.cwrap(B,null,[`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`])}function ux(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,dy:a}=t,{strides:o,pad:s,filterShape:c}=r;if(i.dtype!==`float32`)throw Error(`Tensor dy must have dtype float32, got ${i.dtype}`);if(a.dtype!==`float32`)throw Error(`Tensor filter must have dtype float32, got ${a.dtype}`);let l=Da(i.shape,c,o,1,s),u=n.makeOutput(l.filterShape,a.dtype);return cx(n.dataIdMap.get(i.dataId).id,n.dataIdMap.get(a.dataId).id,n.dataIdMap.get(u.dataId).id,l.batchSize,l.inDepth,l.inHeight,l.inWidth,l.inChannels,l.outDepth,l.outHeight,l.outWidth,l.outChannels,l.strideDepth,l.strideHeight,l.strideWidth,l.dilationDepth,l.dilationHeight,l.dilationWidth,l.filterDepth,l.filterHeight,l.filterWidth,l.padInfo.front,l.padInfo.top,l.padInfo.left),u}let dx={kernelName:B,backendName:`wasm`,setupFunc:lx,kernelFunc:ux},fx
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;function px(e){fx=e.wasm.cwrap(Ge,null,[`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`])}function mx(e){let{inputs:t,backend:n,attrs:r}=e,{dy:i,filter:a}=t,{pad:o,strides:s,inputShape:c}=r;if(i.dtype!==`float32`)throw Error(`Tensor dy must have dtype float32, got ${i.dtype}`);if(a.dtype!==`float32`)throw Error(`Tensor filter must have dtype float32, got ${a.dtype}`);let l=Da(c,a.shape,s,1,o),u=n.makeOutput(l.inShape,i.dtype);return fx(n.dataIdMap.get(a.dataId).id,n.dataIdMap.get(i.dataId).id,n.dataIdMap.get(u.dataId).id,l.batchSize,l.inDepth,l.inHeight,l.inWidth,l.inChannels,l.outDepth,l.outHeight,l.outWidth,l.outChannels,l.strideDepth,l.strideHeight,l.strideWidth,l.dilationDepth,l.dilationHeight,l.dilationWidth,l.filterDepth,l.filterHeight,l.filterWidth,l.padInfo.front,l.padInfo.top,l.padInfo.left),u}let hx={kernelName:Ge,backendName:`wasm`,setupFunc:px,kernelFunc:mx},gx=Z(`Cos`),_x=Z(Ke)
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2021 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
var vx;(function(e){e[e.bilinear=0]=`bilinear`,e[e.nearest=1]=`nearest`})(vx||={});let yx;function bx(e){yx=e.wasm.cwrap(Ye,null,[`number`,`number`,`number`,`number`,`array`,`number`,`number`,`number`,`number`,`number`])}function xx(e){let{backend:t,inputs:n,attrs:r}=e,{method:i,extrapolationValue:a,cropSize:o}=r,{image:s,boxes:c,boxInd:l}=n,u=c.shape[0],[d,f]=o,p=[u,d,f,s.shape[3]],m=t.dataIdMap.get(s.dataId),h;s.dtype!==`float32`&&(h=Vb({backend:t,inputs:{x:s},attrs:{dtype:`float32`}}),m=t.dataIdMap.get(h.dataId));let g=m.id,_=t.dataIdMap.get(c.dataId).id,v=t.dataIdMap.get(l.dataId).id,y=t.makeOutput(p,`float32`),b=t.dataIdMap.get(y.dataId).id,x=new Uint8Array(new Int32Array(s.shape).buffer);return yx(g,_,v,u,x,d,f,vx[i],a,b),h!=null&&t.disposeData(h.dataId),y}let Sx={kernelName:Ye,backendName:`wasm`,setupFunc:bx,kernelFunc:xx},Cx
/**
* @license
* Copyright 2022 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;function wx(e){Cx=e.wasm.cwrap(qe,null,[`number`,`number`,`number`,`number`,`number`,`number`])}function Tx(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{axis:a,exclusive:o,reverse:s}=r,c=i.shape.length;m(i.dtype===`float32`||i.dtype===`int32`,()=>`cumprod does not support ${i.dtype} tensors in the WASM backend`);let l=uo([a],c),u=i;l!==null&&(u=Ry({inputs:{x:i},attrs:{perm:l},backend:n}));let d=po(1,c)[0];lo(`cumprod`,[d],c);let f=n.makeOutput(u.shape,u.dtype),p=u.shape[d],h=n.dataIdMap.get(u.dataId).id,g=n.dataIdMap.get(f.dataId).id;Cx(h,+!!o,+!!s,p,g,X[i.dtype]);let _=f;if(l!==null){let e=fo(l);_=Ry({inputs:{x:f},attrs:{perm:e},backend:n}),n.disposeData(u.dataId),n.disposeData(f.dataId)}return _}let Ex={kernelName:qe,backendName:`wasm`,setupFunc:wx,kernelFunc:Tx},Dx
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;function Ox(e){Dx=e.wasm.cwrap(Je,null,[`number`,`number`,`number`,`number`,`number`,`number`])}function kx(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{axis:a,exclusive:o,reverse:s}=r,c=i.shape.length;m(i.dtype===`float32`||i.dtype===`int32`,()=>`cumsum does not support ${i.dtype} tensors in the WASM backend`);let l=uo([a],c),u=i;l!==null&&(u=Ry({inputs:{x:i},attrs:{perm:l},backend:n}));let d=po(1,c)[0];lo(`cumsum`,[d],c);let f=n.makeOutput(u.shape,u.dtype),p=u.shape[d],h=n.dataIdMap.get(u.dataId).id,g=n.dataIdMap.get(f.dataId).id;Dx(h,+!!o,+!!s,p,g,X[i.dtype]);let _=f;if(l!==null){let e=fo(l);_=Ry({inputs:{x:f},attrs:{perm:e},backend:n}),n.disposeData(u.dataId),n.disposeData(f.dataId)}return _}let Ax={kernelName:Je,backendName:`wasm`,setupFunc:Ox,kernelFunc:kx},jx
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;function Mx(e){jx=e.wasm.cwrap(`DenseBincount`,null,[`number`,`array`,`number`,`number`,`boolean`,`number`,`number`,`boolean`,`number`])}function Nx(e){let{backend:t,inputs:n,attrs:r}=e,{x:i,weights:a}=n,{size:o,binaryOutput:s}=r,c=a.shape.reduce((e,t)=>e*t,1)!==0,l=i.shape.length===1?[o]:[i.shape[0],o],u=t.makeOutput(l,a.dtype);function d(e){return t.dataIdMap.get(e.dataId).id}return jx(d(i),new Uint8Array(new Int32Array(i.shape).buffer),i.shape.length,o,c,d(a),X[a.dtype],s,d(u)),u}let Px={kernelName:Xe,backendName:`wasm`,setupFunc:Mx,kernelFunc:Nx},Fx
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;function Ix(e){Fx=e.wasm.cwrap(Ze,null,[`number`,`number`,`number`,`array`,`number`,`array`,`array`,`number`,`number`])}function Lx(e){let{backend:t,inputs:n,attrs:r}=e,{x:i}=n,{blockSize:a,dataFormat:o}=r,s=i.shape[0],c=o===`NHWC`?i.shape[1]:i.shape[2],l=o===`NHWC`?i.shape[2]:i.shape[3],u=o===`NHWC`?i.shape[3]:i.shape[1],d=c*a,f=l*a,p=u/(a*a),m=o===`NHWC`?[s,d,f,p]:[s,p,d,f],h=t.makeOutput(m,`float32`),g=t.dataIdMap.get(i.dataId).id,_=new Uint8Array(new Int32Array(P(i.shape)).buffer),v=new Uint8Array(new Int32Array(m).buffer),y=new Uint8Array(new Int32Array(P(m)).buffer),b=t.dataIdMap.get(h.dataId).id;return Fx(g,a,+(o===`NHWC`),_,i.shape.length-1,v,y,m.length,b),h}let Rx={kernelName:Ze,backendName:`wasm`,setupFunc:Ix,kernelFunc:Lx},zx
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;function Bx(e){zx=e.wasm.cwrap(Qe,null,[`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`])}function Vx(e){let{inputs:t,attrs:n,backend:r}=e,{x:i,filter:a}=t,o=r.dataIdMap.get(i.dataId).id,s=r.dataIdMap.get(a.dataId).id,{strides:c,dilations:l,pad:u,dimRoundingMode:d}=n,f=l??[1,1],p=Ea(i.shape,a.shape,c,f,u,d,!0),m=p.filterHeight,h=p.filterWidth,g=p.padInfo.top,_=p.padInfo.right,v=p.padInfo.bottom,y=p.padInfo.left,b=p.dilationHeight,x=p.dilationWidth,S=p.strideHeight,C=p.strideWidth,w=p.inChannels,T=p.outChannels,E=+(p.padInfo.type===`SAME`);if(p.dataFormat!==`channelsLast`)throw Error(`wasm backend DepthwiseConv2dNative does not support dataFormat:'${p.dataFormat}'. Please use 'channelsLast'.`);let D=r.makeOutput(p.outShape,`float32`),O=r.dataIdMap.get(D.dataId).id;return zx(o,i.shape[0],i.shape[1],i.shape[2],s,m,h,g,_,v,y,E,b,x,S,C,w,T,O),D}let Hx={kernelName:Qe,backendName:`wasm`,setupFunc:Bx,kernelFunc:Vx},Ux
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;function Wx(e){Ux=e.wasm.cwrap(`Diag`,null,[`number`,`number`,`number`,`number`])}function Gx(e){let{inputs:t,backend:n}=e,{x:r}=t,i=_(r.shape),a=n.makeOutput([...r.shape,...r.shape],r.dtype);return Ux(n.dataIdMap.get(r.dataId).id,X[r.dtype],i,n.dataIdMap.get(a.dataId).id),a}let Kx={kernelName:$e,backendName:`wasm`,setupFunc:Wx,kernelFunc:Gx},qx
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;function Jx(e){qx=e.wasm.cwrap(et,null,[`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`])}function Yx(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,filter:a}=t,{strides:o,pad:s,dilations:c}=r;if(i.dtype!==a.dtype)throw Error(`Dilation2D error: x must have the same dtype as filter. Got ${i.dtype} and ${a.dtype}`);let l=Ca(i.shape,a.shape,o,s,`NHWC`,c),u=n.makeOutput(l.outShape,i.dtype);return qx(n.dataIdMap.get(i.dataId).id,n.dataIdMap.get(a.dataId).id,n.dataIdMap.get(u.dataId).id,X[i.dtype],l.batchSize,l.inChannels,l.inHeight,l.inWidth,l.outHeight,l.outWidth,l.strideHeight,l.strideWidth,l.dilationHeight,l.dilationWidth,l.filterHeight,l.filterWidth,l.padInfo.top,l.padInfo.left),u}let Xx={kernelName:et,backendName:`wasm`,setupFunc:Jx,kernelFunc:Yx},Zx
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;function Qx(e){Zx=e.wasm.cwrap(nt,null,[`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`])}function $x(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,filter:a,dy:o}=t,{strides:s,pad:c,dilations:l}=r;if(i.dtype!==a.dtype||i.dtype!==o.dtype)throw Error(`Dilation2DBackpropFilter error: x must have the same dtype as filter and dy. Got ${i.dtype}, ${a.dtype}, and ${o.dtype}`);let u=Ca(i.shape,a.shape,s,c,`NHWC`,l),d=n.makeOutput(a.shape,a.dtype);return Zx(n.dataIdMap.get(i.dataId).id,n.dataIdMap.get(a.dataId).id,n.dataIdMap.get(o.dataId).id,n.dataIdMap.get(d.dataId).id,X[i.dtype],u.batchSize,u.inChannels,u.inHeight,u.inWidth,u.outHeight,u.outWidth,u.strideHeight,u.strideWidth,u.dilationHeight,u.dilationWidth,u.filterHeight,u.filterWidth,u.padInfo.top,u.padInfo.left),d}let eS={kernelName:nt,backendName:`wasm`,setupFunc:Qx,kernelFunc:$x},tS
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;function nS(e){tS=e.wasm.cwrap(tt,null,[`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`])}function rS(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,filter:a,dy:o}=t,{strides:s,pad:c,dilations:l}=r;if(i.dtype!==a.dtype||i.dtype!==o.dtype)throw Error(`Dilation2DBackpropInput error: x must have the same dtype as filter and dy. Got ${i.dtype}, ${a.dtype}, and ${o.dtype}`);let u=Ca(i.shape,a.shape,s,c,`NHWC`,l),d=n.makeOutput(i.shape,i.dtype);return tS(n.dataIdMap.get(i.dataId).id,n.dataIdMap.get(a.dataId).id,n.dataIdMap.get(o.dataId).id,n.dataIdMap.get(d.dataId).id,X[i.dtype],u.batchSize,u.inChannels,u.inHeight,u.inWidth,u.outHeight,u.outWidth,u.strideHeight,u.strideWidth,u.dilationHeight,u.dilationWidth,u.filterHeight,u.filterWidth,u.padInfo.top,u.padInfo.left),d}let iS={kernelName:tt,backendName:`wasm`,setupFunc:nS,kernelFunc:rS},aS=Z(`Elu`),oS
/**
* @license
* Copyright 2021 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;function sS(e){oS=e.wasm.cwrap(it,null,[`number`,`number`,`number`])}function cS(e){let{inputs:t,backend:n}=e,{dy:r,y:i}=t,a=n.makeOutput(i.shape,`float32`),o=e=>n.dataIdMap.get(e.dataId).id;return oS(o(i),o(r),o(a)),a}let lS={kernelName:it,backendName:`wasm`,setupFunc:sS,kernelFunc:cS},uS=Oy(at,!1,`bool`),dS=Z(`Erf`),fS=Z(`Exp`,`float32`)
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function pS(e){let{inputs:t,attrs:n,backend:r}=e,{input:i}=t,{dim:a}=n,o=i.shape.length,s=i.shape.slice(),c=a;return a<0&&(m(-(o+1)<=a,()=>`Axis must be in the interval [${-(o+1)}, ${o}]`),c=o+a+1),s.splice(c,0,1),xb({inputs:{x:i},backend:r,attrs:{shape:s}})}let mS={kernelName:ot,backendName:`wasm`,kernelFunc:pS},hS=Z(st,`float32`)
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function gS(e){let{attrs:{shape:t,value:n},backend:r}=e,{attrs:{dtype:i}}=e;i||=N(n);let a=r.makeOutput(t,i);return r.typedArrayFromHeap(a).fill(n),a}let _S={kernelName:ct,backendName:`wasm`,kernelFunc:gS},vS
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;function yS(e){vS=e.wasm.cwrap(lt,null,[`number`,`number`,`number`,`number`,`number`,`number`])}function bS(e){let{inputs:t,backend:n}=e,{image:r}=t,i=n.makeOutput(r.shape,r.dtype),a=n.dataIdMap.get(r.dataId).id,o=n.dataIdMap.get(i.dataId).id,[s,c,l,u]=r.shape;return vS(a,s,c,l,u,o),i}let xS={kernelName:lt,backendName:`wasm`,kernelFunc:bS,setupFunc:yS},SS=Z(ut),CS=Oy(dt,!1),wS
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;function TS(e){wS=e.wasm.cwrap(ft,null,[`number`,`number`,`number`,`number`,`number`,`number`,`number`])}function ES(e){let{backend:t,inputs:n,attrs:r}=e,{varianceEpsilon:i}=r,{x:a,mean:o,variance:s,offset:c,scale:l}=n,u=t.dataIdMap.get(a.dataId).id,d=t.dataIdMap.get(o.dataId).id,f=t.dataIdMap.get(s.dataId).id,p=c==null?0:t.dataIdMap.get(c.dataId).id,m=l==null?0:t.dataIdMap.get(l.dataId).id,h=t.makeOutput(a.shape,a.dtype);if(_(a.shape)===0)return h;let g=t.dataIdMap.get(h.dataId).id;return wS(u,d,f,p,m,i,g),h}let DS={kernelName:ft,backendName:`wasm`,setupFunc:TS,kernelFunc:ES},OS
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;function kS(e){OS=e.wasm.cwrap(qn,null,[`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`])}function AS(e){let{inputs:t,attrs:n,backend:r}=e,{x:i,filter:a,bias:o,preluActivationWeights:s}=t,{strides:c,pad:l,dilations:u,dataFormat:d,dimRoundingMode:f,activation:p,leakyreluAlpha:m}=n,h=Ea(i.shape,a.shape,c,u,l,f),g=by[p];if(g==null)throw Error(`${p} activation not yet supported for FusedConv2D in the wasm backend.`);let _=r.dataIdMap.get(i.dataId).id,v=r.dataIdMap.get(a.dataId).id,y=h.outChannels,b=0;if(o!=null){let e=r.dataIdMap.get(o.dataId);if(e.shape.length!==1)throw Error(`FusedConv2D only supports rank-1 bias but got rank ${e.shape.length}.`);if(e.shape[0]!==y)throw Error(`FusedConv2D bias shape (${e.shape}) does not match the number of output channels (${y})`);b=e.id}let x=h.filterHeight,S=h.filterWidth,C=h.padInfo.top,w=h.padInfo.right,T=h.padInfo.bottom,E=h.padInfo.left,D=h.dilationHeight,O=h.dilationWidth,k=h.strideHeight,A=h.strideWidth,j=h.inChannels,M=+(h.padInfo.type===`SAME`),N=h.batchSize,ee=h.inHeight,P=h.inWidth;if(d!==`NHWC`)throw Error(`wasm backend FusedConv2D does not support dataFormat:'${d}'. Please use 'NHWC'.`);let te=r.makeOutput(h.outShape,`float32`),F=r.dataIdMap.get(te.dataId).id,ne=s==null?0:r.dataIdMap.get(s.dataId).id;return OS(_,N,ee,P,v,x,S,b,C,w,T,E,M,D,O,k,A,j,y,g,ne,m||0,F),te}let jS={kernelName:qn,backendName:`wasm`,setupFunc:kS,kernelFunc:AS},MS
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;function NS(e){MS=e.wasm.cwrap(Jn,null,[`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`])}function PS(e){let{inputs:t,attrs:n,backend:r}=e,{x:i,filter:a,bias:o,preluActivationWeights:s}=t,{strides:c,pad:l,dilations:u,dataFormat:d,dimRoundingMode:f,activation:p,leakyreluAlpha:m}=n,h=Ea(i.shape,a.shape,c,u,l,f,!0),g=by[p];if(g==null)throw Error(`${p} activation not yet supported for FusedDepthwiseConv2D in the wasm backend.`);let _=r.dataIdMap.get(i.dataId).id,v=r.dataIdMap.get(a.dataId).id,y=h.outChannels,b=0;if(o!=null){let e=r.dataIdMap.get(o.dataId);if(e.shape.length!==1)throw Error(`FusedDepthwiseConv2D only supports rank-1 bias but got rank ${e.shape.length}.`);if(e.shape[0]!==y)throw Error(`FusedDepthwiseConv2D bias shape (${e.shape}) does not match the number of output channels (${y})`);b=e.id}let x=h.filterHeight,S=h.filterWidth,C=h.padInfo.top,w=h.padInfo.right,T=h.padInfo.bottom,E=h.padInfo.left,D=h.dilationHeight,O=h.dilationWidth,k=h.strideHeight,A=h.strideWidth,j=h.inChannels,M=+(h.padInfo.type===`SAME`),N=h.batchSize,ee=h.inHeight,P=h.inWidth;if(d!==`NHWC`)throw Error(`wasm backend FusedDepthwiseConv2D does not support dataFormat:'${d}'. Please use 'NHWC'.`);let te=r.makeOutput(h.outShape,`float32`),F=r.dataIdMap.get(te.dataId).id,ne=s==null?0:r.dataIdMap.get(s.dataId).id;return MS(_,N,ee,P,v,x,S,b,C,w,T,E,M,D,O,k,A,j,y,g,ne,m||0,F),te}let FS={kernelName:Jn,backendName:`wasm`,setupFunc:NS,kernelFunc:PS},IS
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;function LS(e){IS=e.wasm.cwrap(mt,null,[`number`,`number`,`number`,`number`,`number`,`number`,`array`,`number`])}function RS(e){let{backend:t,inputs:n}=e,{params:r,indices:i}=n,[a,o,s,c]=gs(r,i),l=t.makeOutput(a,r.dtype);if(o===0)return l;let u=i.shape,d=u[u.length-1],f=t.dataIdMap.get(r.dataId).id,p=t.dataIdMap.get(i.dataId).id,m=new Uint8Array(new Int32Array(c).buffer),h=t.dataIdMap.get(l.dataId).id;return IS(f,X[r.dtype],p,o,d,s,m,h),l}let zS={kernelName:mt,backendName:`wasm`,setupFunc:LS,kernelFunc:RS},BS
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;function VS(e){BS=e.wasm.cwrap(`Gather`,null,[`number`,`number`,`array`,`number`,`number`,`number`,`array`,`number`])}function HS(e){let{backend:t,inputs:n,attrs:r}=e,{x:i,indices:a}=n,{axis:o,batchDims:s}=r,c=S(o,i.shape)[0],l=t.readSync(a.dataId),u=i.shape[c];for(let e=0;e<l.length;++e){let t=l[e];m(t<=u-1&&t>=0,()=>`GatherV2: the index value ${t} is not in [0, ${u-1}]`)}let d=fc(i,a,c,s),f=xb({inputs:{x:i},attrs:{shape:[d.batchSize,d.outerSize,d.dimSize,d.sliceSize]},backend:t}),p=_(a.shape),h=xb({inputs:{x:a},attrs:{shape:[d.batchSize,p/d.batchSize]},backend:t}),g=[d.batchSize,d.outerSize,p/d.batchSize,d.sliceSize],v=t.makeOutput(g,i.dtype);if(_(i.shape)===0)return v;let y=f.shape.length-1,b=t.dataIdMap.get(f.dataId).id,x=t.dataIdMap.get(h.dataId).id,C=t.dataIdMap.get(v.dataId).id,w=new Uint8Array(new Int32Array(P(f.shape)).buffer),T=new Uint8Array(new Int32Array(P(g)).buffer);return BS(b,X[i.dtype],w,y,x,d.batchSize,T,C),t.disposeData(f.dataId),t.disposeData(h.dataId),v.shape=d.outputShape,v}let US={kernelName:pt,backendName:`wasm`,setupFunc:VS,kernelFunc:HS},WS=Oy(ht,!1,`bool`),GS=Oy(gt,!1,`bool`),KS=Z(vt,`bool`),qS=Z(yt,`bool`),JS=Z(bt,`bool`),YS
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2022 The TensorFlow Authors. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the License);
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an AS IS BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;function XS(e){YS=e.wasm.cwrap(xt,null,[`number`,`number`,`number`,`number`])}function ZS(e){let{inputs:{x:t},attrs:{alpha:n},backend:r}=e,i=r.dataIdMap.get(t.dataId).id,a=r.makeOutput(t.shape,`float32`);if(_(t.shape)!==0){let e=r.dataIdMap.get(a.dataId).id;YS(i,X[t.dtype],n,e)}return a}let QS={kernelName:xt,backendName:`wasm`,setupFunc:XS,kernelFunc:ZS},$S=Oy(St,!1,`bool`),eC=Oy(Ct,!1,`bool`),tC
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2023 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;function nC(e){tC=e.wasm.cwrap(wt,null,[`number`,`number`,`number`,`number`])}function rC(e){let{attrs:t,backend:n}=e,{start:r,stop:i,num:a}=t,o=Math.floor(a),s=n.makeOutput([o],`float32`);return tC(n.dataIdMap.get(s.dataId).id,r,i,o),s}let iC={kernelName:wt,backendName:`wasm`,setupFunc:nC,kernelFunc:rC},aC=Z(`Log`),oC=Z(Tt),sC=Oy(Et,!1,`bool`),cC=Z(Dt),lC=Oy(Ot,!1,`bool`),uC=Oy(`LogicalXor`,!1,`bool`),dC
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;function fC(e){dC=e.wasm.cwrap(`LRN`,null,[`number`,`number`,`number`,`number`,`number`,`number`,`number`])}function pC(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{depthRadius:a,bias:o,alpha:s,beta:c}=r;if(i.dtype!==`float32`)throw Error(`LRN error: x must have dtype float32`);let l=n.makeOutput(i.shape,i.dtype);return dC(n.dataIdMap.get(i.dataId).id,n.dataIdMap.get(l.dataId).id,i.shape[3],a,o,s,c),l}let mC={kernelName:`LRN`,backendName:`wasm`,setupFunc:fC,kernelFunc:pC},hC
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;function gC(e){hC=e.wasm.cwrap(kt,null,[`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`])}function _C(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,y:a,dy:o}=t,{depthRadius:s,bias:c,alpha:l,beta:u}=r;if(i.dtype!==`float32`||a.dtype!==`float32`||o.dtype!==`float32`)throw Error(`LRNGrad error: x, y, and dy must have dtype float32`);let d=n.makeOutput(i.shape,i.dtype);return hC(n.dataIdMap.get(i.dataId).id,n.dataIdMap.get(a.dataId).id,n.dataIdMap.get(o.dataId).id,n.dataIdMap.get(d.dataId).id,o.shape[3],s,c,l,u),d}let vC={kernelName:kt,backendName:`wasm`,setupFunc:gC,kernelFunc:_C},yC
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;function bC(e){yC=e.wasm.cwrap(`Max`,null,[`number`,`number`,`number`,`number`])}function xC(e){let{backend:t,inputs:n,attrs:r}=e,{reductionIndices:i,keepDims:a}=r,{x:o}=n,s=t.dataIdMap.get(o.dataId).id,c=o,{transposed:l,axes:u,originalAxes:d,inputWasTransposed:f}=Hy(o,i,t);if(f){let e=t.dataIdMap.get(l.dataId).id;c=l,s=e}let p=c.shape.length;lo(`max`,u,p);let[m,h]=so(c.shape,u),g=_(h),v=t.makeOutput(m,o.dtype);if(_(c.shape)!==0){let e=t.dataIdMap.get(v.dataId).id;yC(s,X[o.dtype],g,e)}return f&&t.disposeData(l.dataId),a&&(v.shape=co(v.shape,d)),v}let SC={kernelName:`Max`,backendName:`wasm`,setupFunc:bC,kernelFunc:xC},CC=Oy(At,!1),wC
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;function TC(e){wC=e.wasm.cwrap(jt,null,[`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`])}function EC(e){let{inputs:t,attrs:n,backend:r}=e,i=t.x,a=r.dataIdMap.get(i.dataId).id;m(i.dtype===`float32`,()=>`Error in MaxPool: only float32 input is supported. Got ${i.dtype}.`);let{filterSize:o,strides:s,pad:c,dimRoundingMode:l}=n,u=wa(i.shape,o,s,1,c,l),d=u.filterHeight,f=u.filterWidth,p=u.padInfo.top,h=u.padInfo.right,g=u.padInfo.bottom,_=u.padInfo.left,v=u.dilationHeight,y=u.dilationWidth,b=u.strideHeight,x=u.strideWidth,S=u.inChannels,C=u.outChannels;if(u.dataFormat!==`channelsLast`)throw Error(`wasm backend does not support dataFormat:'${u.dataFormat}'. Please use 'channelsLast'.`);let w=r.makeOutput(u.outShape,`float32`),T=r.dataIdMap.get(w.dataId).id;return wC(a,i.shape[0],i.shape[1],i.shape[2],d,f,p,h,g,_,v,y,b,x,S,C,T),w}let DC={kernelName:jt,backendName:`wasm`,setupFunc:TC,kernelFunc:EC},OC
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;function kC(e){OC=e.wasm.cwrap(`MaxPool3D`,null,[`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`])}function AC(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{filterSize:a,strides:o,pad:s,dimRoundingMode:c,dataFormat:l}=r,u=Ta(i.shape,a,o,1,s,c,l),d=n.makeOutput(u.outShape,i.dtype);return OC(n.dataIdMap.get(i.dataId).id,n.dataIdMap.get(d.dataId).id,u.batchSize,u.inChannels,u.inDepth,u.inHeight,u.inWidth,u.outDepth,u.outHeight,u.outWidth,u.strideDepth,u.strideHeight,u.strideWidth,u.dilationDepth,u.dilationHeight,u.dilationWidth,u.effectiveFilterDepth,u.effectiveFilterHeight,u.effectiveFilterWidth,u.padInfo.front,u.padInfo.top,u.padInfo.left),d}let jC={kernelName:Nt,backendName:`wasm`,setupFunc:kC,kernelFunc:AC},MC
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;function NC(e){MC=e.wasm.cwrap(`MaxPool3DGrad`,null,[`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`])}function PC(e){let{inputs:t,backend:n,attrs:r}=e,{dy:i,input:a}=t,{filterSize:o,strides:s,pad:c,dimRoundingMode:l}=r,u=Ta(a.shape,o,s,1,c,l),d=n.makeOutput(a.shape,a.dtype);return MC(n.dataIdMap.get(a.dataId).id,n.dataIdMap.get(i.dataId).id,n.dataIdMap.get(d.dataId).id,u.batchSize,u.inChannels,u.inDepth,u.inHeight,u.inWidth,u.outDepth,u.outHeight,u.outWidth,u.strideDepth,u.strideHeight,u.strideWidth,u.dilationDepth,u.dilationHeight,u.dilationWidth,u.effectiveFilterDepth,u.effectiveFilterHeight,u.effectiveFilterWidth,u.padInfo.front,u.padInfo.top,u.padInfo.left),d}let FC={kernelName:Pt,backendName:`wasm`,setupFunc:NC,kernelFunc:PC},IC
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;function LC(e){IC=e.wasm.cwrap(`MaxPoolGrad`,null,[`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`])}function RC(e){let{inputs:t,backend:n,attrs:r}=e,{dy:i,input:a}=t,{filterSize:o,strides:s,pad:c,dimRoundingMode:l}=r,u=wa(a.shape,o,s,1,c,l),d=n.makeOutput(a.shape,a.dtype);return IC(n.dataIdMap.get(a.dataId).id,n.dataIdMap.get(i.dataId).id,n.dataIdMap.get(d.dataId).id,u.batchSize,u.inChannels,u.inHeight,u.inWidth,u.outHeight,u.outWidth,u.strideHeight,u.strideWidth,u.dilationHeight,u.dilationWidth,u.effectiveFilterHeight,u.effectiveFilterWidth,u.padInfo.top,u.padInfo.left),d}let zC={kernelName:Mt,backendName:`wasm`,setupFunc:LC,kernelFunc:RC},BC
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;function VC(e){BC=e.wasm.cwrap(`MaxPoolWithArgmax`,null,[`number`,`number`,`number`,`number`,`boolean`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`])}function HC(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{filterSize:a,strides:o,pad:s,includeBatchInIndex:c}=r;m(i.shape.length===4,()=>`Error in maxPool: input must be rank 4 but got rank ${i.shape.length}.`);let l=[1,1];m(Ra(o,l),()=>`Error in maxPool: Either strides or dilations must be 1. Got strides ${o} and dilations '${l}'`);let u=wa(i.shape,a,o,[1,1],s),d=n.makeOutput(u.outShape,i.dtype),f=n.makeOutput(u.outShape,`int32`);return BC(n.dataIdMap.get(i.dataId).id,n.dataIdMap.get(d.dataId).id,n.dataIdMap.get(f.dataId).id,X[i.dtype],c,u.batchSize,u.inChannels,u.inHeight,u.inWidth,u.outHeight,u.outWidth,u.strideHeight,u.strideWidth,u.dilationHeight,u.dilationWidth,u.effectiveFilterHeight,u.effectiveFilterWidth,u.padInfo.top,u.padInfo.left),[d,f]}let UC={kernelName:Ft,backendName:`wasm`,setupFunc:VC,kernelFunc:HC},WC
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;function GC(e){WC=e.wasm.cwrap(It,null,[`number, number, number`])}function KC(e){let{backend:t,inputs:n,attrs:r}=e,{axis:i,keepDims:a}=r,{x:o}=n,s=t.dataIdMap.get(o.dataId).id,c=s,l=o,{transposed:u,axes:d,originalAxes:f,inputWasTransposed:p}=Hy(o,i,t),m=d;if(p){let e=t.dataIdMap.get(u.dataId).id;e!==s&&(l=u,c=e,m=po(m.length,l.shape.length))}lo(`mean`,m,l.shape.length);let[h,g]=so(l.shape,m),v=_(g),y=l;l.dtype!==`float32`&&(y=Vb({backend:t,inputs:{x:l},attrs:{dtype:`float32`}}),c=t.dataIdMap.get(y.dataId).id);let b=t.makeOutput(h,`float32`);if(_(l.shape)!==0){let e=t.dataIdMap.get(b.dataId).id;WC(c,v,e)}return p&&t.disposeData(u.dataId),a&&(b.shape=co(b.shape,f)),l.dtype!==`float32`&&t.disposeData(y.dataId),b}let qC={kernelName:It,backendName:`wasm`,setupFunc:GC,kernelFunc:KC},JC
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;function YC(e){JC=e.wasm.cwrap(`Min`,null,[`number`,`number`,`number`,`number`])}function XC(e){let{backend:t,inputs:n,attrs:r}=e,{axis:i,keepDims:a}=r,{x:o}=n,s=t.dataIdMap.get(o.dataId).id,c=s,l=o,{transposed:u,axes:d,originalAxes:f,inputWasTransposed:p}=Hy(o,i,t);if(p){let e=t.dataIdMap.get(u.dataId).id;e!==s&&(l=u,c=e)}let m=l.shape.length;lo(`min`,d,m);let[h,g]=so(l.shape,d),v=_(g),y=t.makeOutput(h,l.dtype);if(_(l.shape)!==0){let e=t.dataIdMap.get(y.dataId).id;JC(c,X[o.dtype],v,e)}return p&&t.disposeData(u.dataId),a&&(y.shape=co(y.shape,f)),y}let ZC={kernelName:`Min`,backendName:`wasm`,setupFunc:YC,kernelFunc:XC},QC=Oy(Lt,!1)
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;
/**
* @license
* Copyright 2021 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
var $C;(function(e){e[e.reflect=0]=`reflect`,e[e.symmetric=1]=`symmetric`})($C||={});let ew;function tw(e){ew=e.wasm.cwrap(Rt,null,[`number`,`array`,`number`,`number`,`array`,`array`,`number`,`number`])}function nw(e){let{inputs:{x:t},backend:n,attrs:{paddings:r,mode:i}}=e,a=r.map((e,n)=>e[0]+t.shape[n]+e[1]),o=n.dataIdMap.get(t.dataId).id,s=n.makeOutput(a,t.dtype),c=n.dataIdMap.get(s.dataId).id,l=new Uint8Array(new Int32Array(t.shape).buffer),u=r.map(e=>e[0]),d=r.map(e=>e[1]),f=new Uint8Array(new Int32Array(u).buffer),p=new Uint8Array(new Int32Array(d).buffer);return ew(o,l,t.shape.length,X[t.dtype],f,p,$C[i],c),s}let rw={kernelName:Rt,backendName:`wasm`,kernelFunc:nw,setupFunc:tw},iw
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;function aw(e){iw=e.wasm.cwrap(Cn,null,[`number`,`number`,`number`,`number`])}function ow(e){let{backend:t,inputs:{logits:n},attrs:{dim:r}}=e,i=t.dataIdMap.get(n.dataId).id,a=t.makeOutput(n.shape,n.dtype),o=t.dataIdMap.get(a.dataId).id,s=n.shape[r],c=_(n.shape)/s;return _(a.shape)===0||iw(i,o,s,c),a}let sw={kernelName:Cn,backendName:`wasm`,setupFunc:aw,kernelFunc:ow},cw
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;function lw(e){cw=e.wasm.cwrap(zt,null,[`number`,`number`,`number`,`number`,`number`,`number`])}function uw(e){let{inputs:t,backend:n,attrs:r}=e,{logits:i}=t,{numSamples:a,seed:o,normalized:s}=r;if(i.dtype!==`float32`)throw Error(`Tensor logits must have dtype float32, got ${i.dtype}`);let c=s?i:ow({inputs:{logits:i},backend:n,attrs:{dim:i.shape.length-1}}),[l,u]=c.shape,d=n.makeOutput([l,a],`int32`);return cw(n.dataIdMap.get(c.dataId).id,l,u,a,o,n.dataIdMap.get(d.dataId).id),s||n.disposeData(c.dataId),d}let dw={kernelName:zt,backendName:`wasm`,setupFunc:lw,kernelFunc:uw},fw=Oy(`Mod`,!0),pw=Oy(Bt,!0),mw=Z(`Neg`)
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function hw(e,t){let n=new Int32Array(e.wasm.HEAPU8.buffer,t,4),r=n[0],i=n[1],a=n[2],o=n[3];return e.wasm._free(t),{pSelectedIndices:r,selectedSize:i,pSelectedScores:a,pValidOutputs:o}}
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let gw;function _w(e){gw=e.wasm.cwrap(Ht,`number`,[`number`,`number`,`number`,`number`,`number`])}function vw(e){let{backend:t,inputs:n,attrs:r}=e,{iouThreshold:i,maxOutputSize:a,scoreThreshold:o}=r,{boxes:s,scores:c}=n,l=t.dataIdMap.get(s.dataId).id,u=t.dataIdMap.get(c.dataId).id,{pSelectedIndices:d,selectedSize:f,pSelectedScores:p,pValidOutputs:m}=hw(t,gw(l,u,a,i,o));return t.wasm._free(p),t.wasm._free(m),t.makeOutput([f],`int32`,d)}let yw={kernelName:Ht,backendName:`wasm`,setupFunc:_w,kernelFunc:vw},bw
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;function xw(e){bw=e.wasm.cwrap(Ut,`number`,[`number`,`number`,`number`,`number`,`number`,`bool`])}function Sw(e){let{backend:t,inputs:n,attrs:r}=e,{iouThreshold:i,maxOutputSize:a,scoreThreshold:o,padToMaxOutputSize:s}=r,{boxes:c,scores:l}=n,u=t.dataIdMap.get(c.dataId).id,d=t.dataIdMap.get(l.dataId).id,{pSelectedIndices:f,selectedSize:p,pSelectedScores:m,pValidOutputs:h}=hw(t,bw(u,d,a,i,o,s));return t.wasm._free(m),[t.makeOutput([p],`int32`,f),t.makeOutput([],`int32`,h)]}let Cw={kernelName:Ut,backendName:`wasm`,setupFunc:xw,kernelFunc:Sw},ww
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;function Tw(e){ww=e.wasm.cwrap(Wt,`number`,[`number`,`number`,`number`,`number`,`number`,`number`])}function Ew(e){let{backend:t,inputs:n,attrs:r}=e,{iouThreshold:i,maxOutputSize:a,scoreThreshold:o,softNmsSigma:s}=r,{boxes:c,scores:l}=n,u=t.dataIdMap.get(c.dataId).id,d=t.dataIdMap.get(l.dataId).id,{pSelectedIndices:f,selectedSize:p,pSelectedScores:m,pValidOutputs:h}=hw(t,ww(u,d,a,i,o,s));return t.wasm._free(h),[t.makeOutput([p],`int32`,f),t.makeOutput([p],`float32`,m)]}let Dw={kernelName:Wt,backendName:`wasm`,setupFunc:Tw,kernelFunc:Ew},Ow=Oy(Vt,!1,`bool`),kw
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;function Aw(e){kw=e.wasm.cwrap(Kt,null,[`number`,`number`,`number`,`number`,`number`])}function jw(e){let{inputs:t,backend:n,attrs:r}=e,{indices:i}=t,{dtype:a,depth:o,onValue:s,offValue:c}=r,l=n.makeOutput([...i.shape,o],a),u=n.dataIdMap.get(l.dataId).id,d=n.dataIdMap.get(i.dataId).id;return kw(d,o,s,c,u),l}let Mw={kernelName:Kt,backendName:`wasm`,setupFunc:Aw,kernelFunc:jw};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Nw(e){let{inputs:{x:t},backend:n}=e,r=n.makeOutput(t.shape,t.dtype);return n.typedArrayFromHeap(r).fill(1),r}let Pw={kernelName:Gt,backendName:`wasm`,kernelFunc:Nw};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function Fw(e){let{inputs:t,backend:n,attrs:r}=e,{axis:i}=r;if(t.length===1)return pS({inputs:{input:t[0]},backend:n,attrs:{dim:i}});let a=t[0].shape,o=t[0].dtype;t.forEach(e=>{h(a,e.shape,`All tensors passed to stack must have matching shapes`),m(o===e.dtype,()=>`All tensors passed to stack must have matching dtypes`)});let s=[],c=Jb({inputs:t.map(e=>{let t=pS({inputs:{input:e},backend:n,attrs:{dim:i}});return s.push(t),t}),backend:n,attrs:{axis:i}});return s.forEach(e=>n.disposeData(e.dataId)),c}let Iw={kernelName:qt,backendName:`wasm`,kernelFunc:Fw},Lw
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;function Rw(e){Lw=e.wasm.cwrap(Jt,null,[`number`,`array`,`number`,`number`,`array`,`array`,`number`,`number`])}function zw(e){let{inputs:{x:t},backend:n,attrs:{paddings:r,constantValue:i}}=e,a=r.map((e,n)=>e[0]+t.shape[n]+e[1]);if(_(t.shape)===0)return gS({backend:n,attrs:{shape:a,value:i,dtype:t.dtype}});let o=n.dataIdMap.get(t.dataId).id,s=n.makeOutput(a,t.dtype),c=n.dataIdMap.get(s.dataId).id,l=new Uint8Array(new Int32Array(t.shape).buffer),u=r.map(e=>e[0]),d=r.map(e=>e[1]),f=new Uint8Array(new Int32Array(u).buffer),p=new Uint8Array(new Int32Array(d).buffer);return Lw(o,l,t.shape.length,X[t.dtype],f,p,i,c),s}let Bw={kernelName:Jt,backendName:`wasm`,kernelFunc:zw,setupFunc:Rw},Vw=Oy(`Pow`,!1),Hw
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;function Uw(e){Hw=e.wasm.cwrap(Yt,null,[`number`,`number`,`number`])}function Ww(e){let{inputs:t,backend:n}=e,{x:r,alpha:i}=t,a=n.dataIdMap.get(r.dataId).id,o=n.dataIdMap.get(i.dataId).id,s=a,c=r,l=c;c.dtype!==`float32`&&(l=Vb({backend:n,inputs:{x:r},attrs:{dtype:`float32`}}),s=n.dataIdMap.get(l.dataId).id);let u=n.makeOutput(r.shape,`float32`),d=n.dataIdMap.get(u.dataId).id;return Hw(s,o,d),c.dtype!==`float32`&&n.disposeData(l.dataId),u}let Gw={kernelName:Yt,backendName:`wasm`,setupFunc:Uw,kernelFunc:Ww},Kw
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;function qw(e){Kw=e.wasm.cwrap(Xt,null,[`number`,`number`,`number`,`number`])}function Jw(e){let{backend:t,inputs:n,attrs:r}=e,{axis:i,keepDims:a}=r,{x:o}=n,s=t.dataIdMap.get(o.dataId).id,c=s,l=o,{transposed:u,axes:d,originalAxes:f,inputWasTransposed:p}=Hy(o,i,t),m=d;if(p){let e=t.dataIdMap.get(u.dataId).id;e!==s&&(l=u,c=e,m=po(m.length,l.shape.length))}lo(`prod`,m,l.shape.length);let[h,g]=so(l.shape,m),v=_(g),y=t.makeOutput(h,l.dtype);if(_(l.shape)!==0){let e=t.dataIdMap.get(y.dataId).id;Kw(c,v,X[y.dtype],e)}return p&&t.disposeData(u.dataId),a&&(y.shape=co(y.shape,f)),y}let Yw={kernelName:Xt,backendName:`wasm`,setupFunc:qw,kernelFunc:Jw},Xw={kernelName:Zt,backendName:`wasm`,kernelFunc:e=>{
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let{backend:t,attrs:n}=e,{start:r,stop:i,step:a,dtype:o}=n,s=Pl(r,i,a,o),c=t.makeOutput([s.length],o);return t.typedArrayFromHeap(c).set(s),c}},Zw=Oy(rt,!0),Qw=Z(Qt),$w=Z($t),eT=Z(on),tT
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2022 The TensorFlow Authors. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;function nT(e){tT=e.wasm.cwrap(rn,null,[`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`])}function rT(e){let{backend:t,inputs:n,attrs:r}=e,{images:i}=n,{alignCorners:a,halfPixelCenters:o,size:s}=r,[c,l]=s,[u,d,f,p]=i.shape,m=[u,c,l,p],h=t.dataIdMap.get(i.dataId),g;h.dtype!==`float32`&&(g=Vb({backend:t,inputs:{x:i},attrs:{dtype:`float32`}}),h=t.dataIdMap.get(g.dataId));let v=h.id,y=t.makeOutput(m,`float32`);if(_(i.shape)===0)return y;let b=t.dataIdMap.get(y.dataId).id;return tT(v,u,d,f,p,c,l,+!!a,+!!o,b),g!=null&&t.disposeData(g.dataId),y}let iT={kernelName:rn,backendName:`wasm`,setupFunc:nT,kernelFunc:rT},aT
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;function oT(e){aT=e.wasm.cwrap(an,null,[`number`,`number`,`number`,`array`,`array`,`boolean`])}function sT(e){let{inputs:t,backend:n,attrs:r}=e,{images:i,dy:a}=t,{alignCorners:o}=r,s=n.makeOutput(i.shape,`float32`),c=n.dataIdMap.get(i.dataId),l;return c.dtype!==`float32`&&(l=Vb({backend:n,inputs:{x:i},attrs:{dtype:`float32`}}),c=n.dataIdMap.get(l.dataId)),aT(n.dataIdMap.get(i.dataId).id,n.dataIdMap.get(a.dataId).id,n.dataIdMap.get(s.dataId).id,new Uint8Array(new Int32Array(i.shape).buffer),new Uint8Array(new Int32Array(a.shape).buffer),o),l!=null&&n.disposeData(l.dataId),s}let cT={kernelName:an,backendName:`wasm`,setupFunc:oT,kernelFunc:sT},lT
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the 'License');
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an 'AS IS' BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;function uT(e){lT=e.wasm.cwrap(tn,null,[`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`])}function dT(e){let{backend:t,inputs:n,attrs:r}=e,{images:i}=n,{alignCorners:a,halfPixelCenters:o,size:s}=r,[c,l]=s,[u,d,f,p]=i.shape,m=[u,c,l,p],h=t.makeOutput(m,`float32`);if(_(i.shape)===0)return h;let g=t.dataIdMap.get(i.dataId),v;g.dtype!==`float32`&&(v=Vb({backend:t,inputs:{x:i},attrs:{dtype:`float32`}}),g=t.dataIdMap.get(v.dataId));let y=g.id,b=t.dataIdMap.get(h.dataId).id;return lT(y,u,d,f,p,c,l,+!!a,+!!o,b),v!=null&&t.disposeData(v.dataId),h}let fT={kernelName:tn,backendName:`wasm`,setupFunc:uT,kernelFunc:dT},pT
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;function mT(e){pT=e.wasm.cwrap(nn,null,[`number`,`number`,`number`,`array`,`array`,`boolean`])}function hT(e){let{inputs:t,backend:n,attrs:r}=e,{images:i,dy:a}=t,{alignCorners:o}=r,s=n.makeOutput(i.shape,`float32`),c=n.dataIdMap.get(i.dataId),l;return c.dtype!==`float32`&&(l=Vb({backend:n,inputs:{x:i},attrs:{dtype:`float32`}}),c=n.dataIdMap.get(l.dataId)),pT(n.dataIdMap.get(i.dataId).id,n.dataIdMap.get(a.dataId).id,n.dataIdMap.get(s.dataId).id,new Uint8Array(new Int32Array(i.shape).buffer),new Uint8Array(new Int32Array(a.shape).buffer),o),l!=null&&n.disposeData(l.dataId),s}let gT={kernelName:nn,backendName:`wasm`,setupFunc:mT,kernelFunc:hT},_T
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;function vT(e){_T=e.wasm.cwrap(sn,null,[`number`,`array`,`number`,`array`,`number`,`number`])}function yT(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{dims:a}=r,o=S(a,i.shape);if(i.shape.length===0)return Py({inputs:{x:i},backend:n});let s=n.makeOutput(i.shape,i.dtype),c=n.dataIdMap.get(i.dataId).id,l=n.dataIdMap.get(s.dataId).id,u=new Uint8Array(new Int32Array(o).buffer),d=new Uint8Array(new Int32Array(i.shape).buffer);_T(c,u,o.length,d,i.shape.length,l);let f=xb({inputs:{x:s},attrs:{shape:i.shape},backend:n});return n.disposeData(s.dataId),f}let bT={kernelName:sn,backendName:`wasm`,kernelFunc:yT,setupFunc:vT},xT
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;function ST(e){xT=e.wasm.cwrap(Gn,null,[`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`array`,`number`,`number`])}function CT(e){let{inputs:t,backend:n,attrs:r}=e,{image:i}=t,{radians:a,fillValue:o,center:s}=r,c=n.makeOutput(i.shape,i.dtype),l=n.dataIdMap.get(i.dataId).id,u=n.dataIdMap.get(c.dataId).id,[d,f,p,m]=i.shape,[h,g]=Ms(s,f,p),_=typeof o==`number`?[o,o,o,o===0?0:255]:[...o,255],v=new Uint8Array(new Int32Array(_).buffer);return xT(l,d,f,p,m,a,h,g,v,_.length,u),c}let wT={kernelName:Gn,backendName:`wasm`,kernelFunc:CT,setupFunc:ST},TT=Z(cn),ET=Z(ln),DT
/**
* @license
* Copyright 2021 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;function OT(e){DT=e.wasm.cwrap(un,null,[`number`,`number`,`number`,`number`,`number`,`number`,`array`,`number`,`number`])}function kT(e){let{backend:t,inputs:n,attrs:r}=e,{indices:i,updates:a}=n,{shape:o}=r,s=t.makeOutput(o,a.dtype);if(_(o)===0)return s;let{sliceRank:c,numUpdates:l,sliceSize:u,strides:d,outputSize:f}=Wo(a,i,o),p=t.dataIdMap.get(i.dataId).id,m=t.dataIdMap.get(a.dataId).id,h=new Uint8Array(new Int32Array(d).buffer),g=t.dataIdMap.get(s.dataId).id;return DT(p,m,X[a.dtype],c,l,u,h,f,g),s}let AT={kernelName:un,backendName:`wasm`,setupFunc:OT,kernelFunc:kT},jT
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;function MT(e){jT=e.wasm.cwrap(fn,null,[`number`,`number`,`number`,`number`,`number`,`number`,`bool`,`number`])}function NT(e){let{inputs:t,backend:n,attrs:r}=e,{sortedSequence:i,values:a}=t,{side:o}=r;if(i.dtype!==a.dtype)throw Error(`SearchSorted error: sorted_sequence must have the same dtype as values. Got ${i.dtype} and ${a.dtype}`);let s=n.makeOutput(a.shape,`int32`);function c(e){return n.dataIdMap.get(e.dataId).id}return jT(c(i),c(a),i.shape[0],i.shape[1],a.shape[1],X[i.dtype],o===`left`,c(s)),s}let PT={kernelName:fn,backendName:`wasm`,setupFunc:MT,kernelFunc:NT},FT
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;function IT(e){FT=e.wasm.cwrap(`SelectV2`,null,[`number`,`number`,`number`,`number`,`number`])}function LT(e){let{inputs:t,backend:n}=e,{condition:r,t:i,e:a}=t,o=n.dataIdMap.get(r.dataId).id,s=n.dataIdMap.get(i.dataId).id,c=n.dataIdMap.get(a.dataId).id,l=n.makeOutput(i.shape,i.dtype),u=n.dataIdMap.get(l.dataId).id,d=r.shape.length,f=i.shape.length,p=d===0||d>1||f===1?1:_(i.shape.slice(1));return FT(o,s,c,p,u),l}let RT={kernelName:pn,backendName:`wasm`,kernelFunc:LT,setupFunc:IT},zT=Z(mn),BT
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;function VT(e){BT=e.wasm.cwrap(vn,null,[`number`,`number`])}function HT(e){let{backend:t,inputs:{x:n}}=e,r=t.dataIdMap.get(n.dataId).id,i=t.makeOutput(n.shape,n.dtype),a=t.dataIdMap.get(i.dataId).id;return _(i.shape)===0||BT(r,a),i}let UT={kernelName:`Sigmoid`,backendName:`wasm`,setupFunc:VT,kernelFunc:HT},WT=Z(_n),GT=Z(`Sin`),KT=Z(gn),qT=Z(yn)
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;
/**
* @license
* Copyright 2021 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function JT(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{blockShape:a,paddings:o}=r,s=_(a),c=[[0,0]];c.push(...o);for(let e=1+a.length;e<i.shape.length;++e)c.push([0,0]);let l=Bw.kernelFunc({inputs:{x:i},backend:n,attrs:{paddings:c,constantValue:0}}),u=Ns(l.shape,a,s,!1),d=Ps(u.length,a.length,!1),f=Fs(l.shape,a,s,!1),p=xb({inputs:{x:l},backend:n,attrs:{shape:u}}),m=Ry({inputs:{x:p},backend:n,attrs:{perm:d}}),h=xb({inputs:{x:m},backend:n,attrs:{shape:f}});return n.disposeData(l.dataId),n.disposeData(p.dataId),n.disposeData(m.dataId),h}let YT={kernelName:xn,backendName:`wasm`,kernelFunc:JT},XT
/**
* @license
* Copyright 2021 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;function ZT(e){XT=e.wasm.cwrap(`SparseFillEmptyRows`,`number`,[`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`])}function QT(e){let{backend:t,inputs:n}=e,{indices:r,values:i,denseShape:a,defaultValue:o}=n,s=r.shape[0],c=r.shape[1],l=t.readSync(a.dataId)[0],u=[s+l,c],d=t.dataIdMap.get(r.dataId).id,f=t.dataIdMap.get(i.dataId).id,p=t.dataIdMap.get(o.dataId).id,m=t.makeOutput(u,r.dtype),h=t.dataIdMap.get(m.dataId).id,g=t.makeOutput(u.slice(0,1),i.dtype),_=t.dataIdMap.get(g.dataId).id,v=t.makeOutput([l],`bool`),y=t.dataIdMap.get(v.dataId).id,b=t.makeOutput([s],r.dtype),x=t.dataIdMap.get(b.dataId).id,S=t.makeOutput([4],`int32`),C=t.dataIdMap.get(S.dataId).id,w=XT(d,f,X[i.dtype],s,l,c,p,h,_,y,x,C),T=t.readSync(S.dataId),E;switch(T[0]){case 1:E=ec(T[1]);break;case 2:E=tc(T[1],T[2]);break;case 3:E=nc(T[1],T[2],T[3]);break;default:E=``}if(t.disposeData(S.dataId),E)throw t.disposeData(m.dataId),t.disposeData(g.dataId),t.disposeData(v.dataId),t.disposeData(b.dataId),Error(E);let D=m,O=g;return w!==u[0]&&(D=Db({inputs:{x:m},attrs:{begin:0,size:[w,c]},backend:t}),O=Db({inputs:{x:g},attrs:{begin:0,size:w},backend:t}),t.disposeData(m.dataId),t.disposeData(g.dataId)),[D,O,v,b]}let $T={kernelName:wn,backendName:`wasm`,setupFunc:ZT,kernelFunc:QT},eE
/**
* @license
* Copyright 2021 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;function tE(e){eE=e.wasm.cwrap(Tn,null,[`number`,`number`,`number`,`number`,`number`,`number`,`number`])}function nE(e){let{backend:t,inputs:n}=e,{inputIndices:r,inputShape:i,newShape:a}=n;if(r.shape.length!==2)throw Error(`Input indices should be a matrix but received shape
        ${r.shape}`);if(i.shape.length!==1)throw Error(`Input shape should be a vector but received shape
        ${i.shape}`);if(a.shape.length!==1)throw Error(`Target shape should be a vector but received shape ${a.shape}`);let o=t.dataIdMap.get(r.dataId).id,s=t.dataIdMap.get(i.dataId).id,c=t.dataIdMap.get(a.dataId).id,l=r.shape[0],u=_(a.shape),d=t.makeOutput([l,u],r.dtype),f=t.dataIdMap.get(d.dataId).id,p=t.makeOutput([u],a.dtype),m=t.dataIdMap.get(p.dataId).id,h=t.makeOutput([3],`int32`),g=t.dataIdMap.get(h.dataId).id;eE(o,s,c,l,f,m,g);let v=t.readSync(h.dataId),y;switch(v[0]){case 0:y=rc(v[1],v[2]);break;case 1:y=ic(v[1],v[2]);break;case 2:y=ac();break;case 3:y=oc(Array.from(t.readSync(i.dataId)),Array.from(t.readSync(p.dataId)));break;case 4:y=sc(Array.from(t.readSync(i.dataId)),Array.from(t.readSync(p.dataId)));break;default:y=``}if(t.disposeData(h.dataId),y)throw t.disposeData(d.dataId),t.disposeData(p.dataId),Error(y);return[d,p]}let rE={kernelName:Tn,backendName:`wasm`,setupFunc:tE,kernelFunc:nE},iE
/**
* @license
* Copyright 2021 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;function aE(e){iE=e.wasm.cwrap(`SparseSegmentReduction`,null,[`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`])}function oE(e,t){let{backend:n,inputs:r}=e,{data:i,indices:a,segmentIds:o}=r,s=a.shape[0],c=n.readSync(o.dataId,s-1,s)[0],l=s>0?c+1:0;if(l<0)throw Error(cc());let u=i.shape.slice();u[0]=l;let d=n.dataIdMap.get(i.dataId).id,f=n.dataIdMap.get(a.dataId).id,p=n.dataIdMap.get(o.dataId).id,m=n.makeOutput(u,i.dtype),h=n.dataIdMap.get(m.dataId).id,g=n.makeOutput([4],`int32`),_=n.dataIdMap.get(g.dataId).id;iE(d,X[i.dtype],i.shape[0],f,p,h,_,t,0);let v=n.readSync(g.dataId),y;switch(v[0]){case 0:y=cc();break;case 1:y=lc();break;case 2:y=uc(v[1],v[2]);break;case 3:y=dc(v[1],v[2],v[3]);break;default:y=``}if(n.disposeData(g.dataId),y)throw n.disposeData(m.dataId),Error(y);return m}
/**
* @license
* Copyright 2021 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function sE(e){return oE(e,!0)}let cE={kernelName:En,backendName:`wasm`,setupFunc:aE,kernelFunc:sE};
/**
* @license
* Copyright 2021 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function lE(e){return oE(e,!1)}let uE={kernelName:Dn,backendName:`wasm`,setupFunc:aE,kernelFunc:lE},dE
/**
* @license
* Copyright 2021 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;function fE(e){dE=e.wasm.cwrap(On,null,[`number`,`number`,`number`,`number`,`number`,`number`,`number`,`number`,`array`,`number`,`number`])}function pE(e){let{backend:t,inputs:n,attrs:r}=e,{sparseIndices:i,sparseValues:a,defaultValue:o}=n,{outputShape:s}=r,c=t.makeOutput(s,o.dtype);if(_(s)===0)return c;let{sliceRank:l,numUpdates:u,sliceSize:d,strides:f,outputSize:p}=Wo(a,i,s),m=t.dataIdMap.get(i.dataId).id,h=t.dataIdMap.get(a.dataId).id,g=t.dataIdMap.get(o.dataId).id,v=new Uint8Array(new Int32Array(f).buffer),y=t.dataIdMap.get(c.dataId).id;return dE(m,h,a.shape.length,g,X[o.dtype],l,u,d,v,p,y),c}let mE={kernelName:On,backendName:`wasm`,setupFunc:fE,kernelFunc:pE};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function hE(e){let{inputs:t,attrs:n,backend:r}=e,{x:i}=t,{numOrSizeSplits:a,axis:o}=n,s=S(o,i.shape)[0],c=$s(i,a,s),l=Array(i.shape.length).fill(0),u=i.shape.slice();return c.map(e=>{let t=[...u];t[s]=e;let n=Db({inputs:{x:i},attrs:{begin:l,size:t},backend:r});return l[s]+=e,n})}let gE={kernelName:Sn,backendName:`wasm`,kernelFunc:hE},_E=Z(bn),vE=Z(An),yE=Oy(kn,!0),bE
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;function xE(e){bE=e.wasm.cwrap(Wn,null,[`number`,`number`,`number`,`number`])}function SE(e){let{backend:t,inputs:n,attrs:r}=e,{alpha:i}=r,{x:a}=n,o=t.dataIdMap.get(a.dataId).id,s=t.makeOutput(a.shape,a.dtype),c=t.dataIdMap.get(s.dataId).id;return bE(o,i,X[a.dtype],c),s}let CE={kernelName:Wn,backendName:`wasm`,setupFunc:xE,kernelFunc:SE},wE
/**
* @license
* Copyright 2021 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;function TE(e){wE=e.wasm.cwrap(Mn,null,[`number`,`array`,`number`,`array`,`array`,`array`,`array`,`array`,`number`,`number`])}function EE(e){let{backend:t,inputs:n,attrs:r}=e,{x:i}=n,{begin:a,end:o,strides:s,beginMask:c,endMask:l,ellipsisMask:u,newAxisMask:d,shrinkAxisMask:f}=r,{finalShapeSparse:p,finalShape:h,isIdentity:g,sliceDim0:_,isSimpleSlice:v,begin:y,end:b,strides:x}=Ss(i.shape,a,o,s,c,l,u,d,f),S;if(g)S=xb({inputs:{x:i},backend:t,attrs:{shape:h}});else if(_||v){m(i.shape.length>=1,()=>`Input must have rank at least 1, got: ${i.shape.length}`);let e=vs(y,b,x),n=Db({inputs:{x:i},backend:t,attrs:{begin:y,size:e}});S=xb({inputs:{x:n},backend:t,attrs:{shape:h}}),t.disposeData(n.dataId)}else{let e=t.makeOutput(p,`float32`),n=t.dataIdMap.get(i.dataId).id,r=new Uint8Array(new Int32Array(P(i.shape)).buffer),a=new Uint8Array(new Int32Array(y).buffer),o=new Uint8Array(new Int32Array(b).buffer),s=new Uint8Array(new Int32Array(x).buffer),c=new Uint8Array(new Int32Array(p).buffer),l=new Uint8Array(new Int32Array(P(p)).buffer),u=t.dataIdMap.get(e.dataId).id;wE(n,r,i.shape.length,a,o,s,c,l,p.length,u),S=xb({inputs:{x:e},backend:t,attrs:{shape:h}}),t.disposeData(e.dataId)}return S}let DE={kernelName:Mn,backendName:`wasm`,setupFunc:TE,kernelFunc:EE};
/**
* @license
* Copyright 2021 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function OE(e){let{backend:t,inputs:n,attrs:r}=e,{data:i,dataSplits:a}=n,{separator:o,nGramWidths:s,leftPad:c,rightPad:l,padWidth:u,preserveShortSequences:d}=r,[f,p]=Xl(t.readSync(i.dataId),t.readSync(a.dataId),o,s,c,l,u,d),m=t.makeOutput([f.length],`string`),h=t.dataIdMap.get(m.dataId);h.stringBytes=f;let g=t.makeOutput(a.shape,`int32`);return t.typedArrayFromHeap(g).set(p),[m,g]}let kE={kernelName:Nn,backendName:`wasm`,kernelFunc:OE};
/**
* @license
* Copyright 2021 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function AE(e){let{backend:t,inputs:n,attrs:r}=e,{input:i,delimiter:a}=n,{skipEmpty:o}=r,[s,c,l]=Ql(t.readSync(i.dataId),t.readSync(a.dataId)[0],o),u=c.length,d=t.makeOutput([u,2],`int32`);t.typedArrayFromHeap(d).set(s);let f=t.makeOutput([u],`string`),p=t.dataIdMap.get(f.dataId);p.stringBytes=c;let m=t.makeOutput([2],`int32`);return t.typedArrayFromHeap(m).set(l),[d,f,m]}let jE={kernelName:Pn,backendName:`wasm`,kernelFunc:AE};
/**
* @license
* Copyright 2021 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function ME(e){let{backend:t,inputs:n,attrs:r}=e,{input:i}=n,{numBuckets:a}=r,o=$l(t.readSync(i.dataId),a),s=t.makeOutput(i.shape,`int32`);return t.typedArrayFromHeap(s).set(o),s}let NE={kernelName:Fn,backendName:`wasm`,kernelFunc:ME},PE=Oy(`Sub`,!0),FE
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;function IE(e){FE=e.wasm.cwrap(`Sum`,null,[`number`,`number`,`number`,`number`])}function LE(e){let{backend:t,inputs:n,attrs:r}=e,{axis:i,keepDims:a}=r,{x:o}=n,s=t.dataIdMap.get(o.dataId).id,c=s,l=o,{transposed:u,axes:d,originalAxes:f,inputWasTransposed:p}=Hy(o,i,t),m=d;if(p){let e=t.dataIdMap.get(u.dataId).id;e!==s&&(l=u,c=e,m=po(m.length,l.shape.length))}lo(`sum`,m,l.shape.length);let[h,g]=so(l.shape,m),v=_(g),y=t.makeOutput(h,l.dtype);if(_(l.shape)!==0){let e=t.dataIdMap.get(y.dataId).id;FE(c,v,X[y.dtype],e)}return p&&t.disposeData(u.dataId),a&&(y.shape=co(y.shape,f)),y}let RE={kernelName:`Sum`,backendName:`wasm`,setupFunc:IE,kernelFunc:LE},zE=Z(`Tan`),BE=Z(In),VE
/**
* @license
* Copyright 2021 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
/**
* @license
* Copyright 2022 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;function HE(e){VE=e.wasm.cwrap(dn,null,[`number`,`number`,`number`,`number`,`number`,`number`,`array`,`number`,`number`,`number`])}function UE(e){let{backend:t,inputs:n,attrs:r}=e,{tensor:i,indices:a,updates:o}=n,{}=r,s=t.makeOutput(i.shape,i.dtype);if(_(i.shape)===0)return s;let{sliceRank:c,numUpdates:l,sliceSize:u,strides:d,outputSize:f}=Wo(o,a,i.shape),p=t.dataIdMap.get(a.dataId).id,m=t.dataIdMap.get(o.dataId).id,h=t.dataIdMap.get(i.dataId).id,g=new Uint8Array(new Int32Array(d).buffer),v=t.dataIdMap.get(s.dataId).id;return VE(p,m,X[o.dtype],c,l,u,g,f,v,h),s}let WE={kernelName:dn,backendName:`wasm`,setupFunc:HE,kernelFunc:UE},GE
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;function KE(e){GE=e.wasm.cwrap(Ln,null,[`number`,`array`,`number`,`array`,`number`,`number`])}function qE(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,a=n.dataIdMap.get(i.dataId).id,{reps:o}=r,s=Array(i.shape.length);for(let e=0;e<s.length;e++)s[e]=i.shape[e]*o[e];let c=new Uint8Array(new Int32Array(i.shape).buffer),l=new Uint8Array(new Int32Array(s).buffer),u=n.makeOutput(s,i.dtype),d=n.dataIdMap.get(u.dataId).id;return GE(a,c,i.shape.length,l,s.length,X[u.dtype],d),u}let JE={kernelName:Ln,backendName:`wasm`,setupFunc:KE,kernelFunc:qE},YE
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;function XE(e){YE=e.wasm.cwrap(Rn,null,[`number`,`array`,`number`,`number`,`number`,`bool`,`number`,`number`])}let ZE={kernelName:Rn,backendName:`wasm`,setupFunc:XE,kernelFunc:({inputs:e,backend:t,attrs:n})=>{let{x:r}=e,{k:i,sorted:a}=n,o=t.dataIdMap.get(r.dataId).id,s=new Uint8Array(new Int32Array(r.shape).buffer),c=r.shape.slice();c[c.length-1]=i;let l=t.makeOutput(c,r.dtype),u=t.dataIdMap.get(l.dataId).id,d=t.makeOutput(c,`int32`),f=t.dataIdMap.get(d.dataId).id;return YE(o,s,r.shape.length,X[r.dtype],i,a,u,f),[l,d]}},QE
/**
* @license
* Copyright 2021 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
;function $E(e){QE=e.wasm.cwrap(zn,null,[`number`,`number`,`bool`,`number`,`number`,`number`,`number`,`number`,`number`,`array`,`number`,`array`,`number`,`number`,`number`,`number`,`number`])}function eD(e){let{backend:t,inputs:n,attrs:r}=e,{image:i,transforms:a}=n,{interpolation:o,fillMode:s,fillValue:c,outputShape:l}=r,[u,d,f,p]=i.shape,[m,h]=l??[d,f],g=[u,m,h,p],_=new Uint8Array(new Int32Array(P(i.shape)).buffer),v=new Uint8Array(new Int32Array(P(g)).buffer),y=t.makeOutput(g,i.dtype),b=t.dataIdMap.get(y.dataId).id,x=t.dataIdMap.get(i.dataId).id,S=t.dataIdMap.get(a.dataId).id,C=o===`nearest`?1:2,w;switch(s){case`constant`:w=1;break;case`reflect`:w=2;break;case`wrap`:w=3;break;case`nearest`:w=4;break;default:w=1}return QE(x,S,a.shape[0]>1,u,m,h,p,f,d,_,i.shape.length-1,v,g.length-1,C,w,c,b),y}let tD={kernelName:zn,backendName:`wasm`,setupFunc:$E,kernelFunc:eD};
/**
* @license
* Copyright 2023 Google LLC.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function nD(e){let{inputs:t,attrs:n,backend:r}=e,{axis:i}=n,{x:a}=t,{outputValues:o,outputShape:s,indices:c}=ou(r.readSync(a.dataId),i,a.shape,a.dtype);return[r.makeOutput(s,a.dtype,void 0,o),r.makeOutput([c.length],`int32`,void 0,c)]}let rD={kernelName:Vn,backendName:`wasm`,kernelFunc:nD};
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function iD(e){let{inputs:t,backend:n,attrs:r}=e,{value:i}=t,{axis:a}=r;a<0&&(a+=i.shape.length);let o=i.shape[a],s=i.shape.length,c=Array(s-1),l=0;for(let e=0;e<s;e++)e!==a&&(c[l++]=i.shape[e]);let u=Array(o),d=Array(s).fill(0),f=i.shape.slice();f[a]=1;for(let e=0;e<u.length;e++)d[a]=e,u[e]=Db({inputs:{x:i},attrs:{begin:d,size:f},backend:n});return u.map(({dataId:e,dtype:t})=>({dataId:e,dtype:t,shape:c}))}let aD={kernelName:Hn,backendName:`wasm`,kernelFunc:iD};
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
function oD(e){let{inputs:{x:t},backend:n}=e,r=n.makeOutput(t.shape,t.dtype);return n.typedArrayFromHeap(r).fill(0),r}
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let sD=[wy,Ty,Ey,Dy,ky,Ny,Ky,Xy,Qy,$y,eb,tb,nb,rb,ib,cb,bb,fb,gb,Eb,Nb,Lb,Rb,Bb,Hb,Ub,qb,Yb,$b,rx,sx,dx,hx,gx,_x,Sx,Ex,Ax,Px,Rx,Hx,Kx,Xx,eS,iS,aS,lS,uS,dS,fS,mS,hS,_S,xS,SS,CS,DS,jS,FS,zS,US,WS,GS,Fy,KS,qS,JS,QS,$S,eC,iC,oC,aC,sC,cC,lC,uC,mC,vC,SC,CC,DC,jC,FC,zC,UC,qC,ZC,QC,rw,dw,fw,pw,mw,yw,Cw,Dw,Ow,Mw,Pw,Iw,Bw,Vw,Gw,Yw,Xw,Zw,Qw,$w,eT,Sb,iT,cT,fT,gT,bT,wT,TT,ET,AT,PT,RT,zT,UT,WT,GT,KT,jb,sw,qT,YT,$T,rE,cE,uE,mE,gE,_E,vE,yE,CE,DE,kE,jE,NE,PE,RE,zE,BE,WE,JE,ZE,tD,Vy,rD,aD,{kernelName:Un,backendName:`wasm`,kernelFunc:oD}];for(let e of sD)tr(e);
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let cD=z();cD.registerFlag(`WASM_HAS_SIMD_SUPPORT`,async()=>{try{return WebAssembly.validate(new Uint8Array([0,97,115,109,1,0,0,0,1,4,1,96,0,0,3,2,1,0,10,9,1,7,0,65,0,253,15,26,11]))}catch{return!1}}),cD.registerFlag(`WASM_HAS_MULTITHREAD_SUPPORT`,async()=>{if(cD.get(`IS_NODE`))return!1;try{return new MessageChannel().port1.postMessage(new SharedArrayBuffer(1)),WebAssembly.validate(new Uint8Array([0,97,115,109,1,0,0,0,1,4,1,96,0,0,3,2,1,0,5,4,1,3,1,1,10,11,1,9,0,65,0,254,16,2,0,26,11]))}catch{return!1}});var lD=o(((e,t)=>{var n=(()=>{var e=typeof document<`u`&&document.currentScript?document.currentScript.src:void 0;return typeof __filename<`u`&&(e||=__filename),(function(t){t||={};function n(){return M.buffer!=L&&le(M.buffer),R}function r(){return M.buffer!=L&&le(M.buffer),ae}function i(){return M.buffer!=L&&le(M.buffer),oe}function a(){return M.buffer!=L&&le(M.buffer),se}function o(){return M.buffer!=L&&le(M.buffer),ce}var s=t===void 0?{}:t,c,l;s.ready=new Promise(function(e,t){c=e,l=t});var u;typeof process<`u`&&process.listeners&&(u={uncaughtException:process.listeners(`uncaughtException`),unhandledRejection:process.listeners(`unhandledRejection`)});var d=Object.assign({},s),f=[],p=(e,t)=>{throw t},m=typeof window==`object`,h=typeof importScripts==`function`,g=typeof process==`object`&&typeof process.versions==`object`&&typeof process.versions.node==`string`,_=s.ENVIRONMENT_IS_PTHREAD||!1,v=``;function y(e){return s.locateFile?s.locateFile(e,v):v+e}var b,x,S;function C(e){e instanceof Fe||k(`exiting due to exception: `+e)}if(g){var w=oa(),T=oa();v=h?T.dirname(v)+`/`:__dirname+`/`,b=(e,t)=>(e=ke(e)?new URL(e):T.normalize(e),w.readFileSync(e,t?void 0:`utf8`)),S=e=>{var t=b(e,!0);return t.buffer||(t=new Uint8Array(t)),t},x=(e,t,n)=>{e=ke(e)?new URL(e):T.normalize(e),w.readFile(e,function(e,r){e?n(e):t(r.buffer)})},process.argv.length>1&&process.argv[1].replace(/\\/g,`/`),f=process.argv.slice(2),process.on(`uncaughtException`,function(e){if(!(e instanceof Fe))throw e}),process.on(`unhandledRejection`,function(e){throw e}),p=(e,t)=>{if(me())throw process.exitCode=e,t;C(t),process.exit(e)},s.inspect=function(){return`[Emscripten Module object]`};let e;try{e=oa()}catch(e){throw console.error(`The "worker_threads" module is not supported in this node.js build - perhaps a newer version is needed?`),e}global.Worker=e.Worker}else(m||h)&&(h?v=self.location.href:typeof document<`u`&&document.currentScript&&(v=document.currentScript.src),e!==void 0&&e&&(v=e),v=v.indexOf(`blob:`)===0?``:v.substr(0,v.replace(/[?#].*/,``).lastIndexOf(`/`)+1),g||(b=e=>{var t=new XMLHttpRequest;return t.open(`GET`,e,!1),t.send(null),t.responseText},h&&(S=e=>{var t=new XMLHttpRequest;return t.open(`GET`,e,!1),t.responseType=`arraybuffer`,t.send(null),new Uint8Array(t.response)}),x=(e,t,n)=>{var r=new XMLHttpRequest;r.open(`GET`,e,!0),r.responseType=`arraybuffer`,r.onload=()=>{if(r.status==200||r.status==0&&r.response){t(r.response);return}n()},r.onerror=n,r.send(null)}));g&&typeof performance>`u`&&(global.performance=oa().performance);var E=console.log.bind(console),D=console.warn.bind(console);g&&(E=e=>w.writeSync(1,e+`
`),D=e=>w.writeSync(2,e+`
`));var O=s.print||E,k=s.printErr||D;Object.assign(s,d),d=null,s.arguments&&(f=s.arguments),s.thisProgram&&s.thisProgram,s.quit&&(p=s.quit),Atomics.load,Atomics.store,Atomics.compareExchange;var A;s.wasmBinary&&(A=s.wasmBinary);var j=s.noExitRuntime||!0;typeof WebAssembly!=`object`&&Ee(`no native wasm support detected`);var M,N,ee=!1,P;function te(e,t){e||Ee(t)}var F=typeof TextDecoder<`u`?new TextDecoder(`utf8`):void 0;function ne(e,t,n){t>>>=0;for(var r=t+n,i=t;e[i]&&!(i>=r);)++i;if(i-t>16&&e.buffer&&F)return F.decode(e.buffer instanceof SharedArrayBuffer?e.slice(t,i):e.subarray(t,i));for(var a=``;t<i;){var o=e[t++];if(!(o&128)){a+=String.fromCharCode(o);continue}var s=e[t++]&63;if((o&224)==192){a+=String.fromCharCode((o&31)<<6|s);continue}var c=e[t++]&63;if(o=(o&240)==224?(o&15)<<12|s<<6|c:(o&7)<<18|s<<12|c<<6|e[t++]&63,o<65536)a+=String.fromCharCode(o);else{var l=o-65536;a+=String.fromCharCode(55296|l>>10,56320|l&1023)}}return a}function re(e,t){return e>>>=0,e?ne(r(),e,t):``}function I(e,t,n,r){if(n>>>=0,!(r>0))return 0;for(var i=n,a=n+r-1,o=0;o<e.length;++o){var s=e.charCodeAt(o);if(s>=55296&&s<=57343){var c=e.charCodeAt(++o);s=65536+((s&1023)<<10)|c&1023}if(s<=127){if(n>=a)break;t[n++>>>0]=s}else if(s<=2047){if(n+1>=a)break;t[n++>>>0]=192|s>>6,t[n++>>>0]=128|s&63}else if(s<=65535){if(n+2>=a)break;t[n++>>>0]=224|s>>12,t[n++>>>0]=128|s>>6&63,t[n++>>>0]=128|s&63}else{if(n+3>=a)break;t[n++>>>0]=240|s>>18,t[n++>>>0]=128|s>>12&63,t[n++>>>0]=128|s>>6&63,t[n++>>>0]=128|s&63}}return t[n>>>0]=0,n-i}function ie(e,t,n){return I(e,r(),t,n)}var L,R,ae,oe,se,ce;_&&(L=s.buffer);function le(e){L=e,s.HEAP8=R=new Int8Array(e),s.HEAP16=new Int16Array(e),s.HEAP32=oe=new Int32Array(e),s.HEAPU8=ae=new Uint8Array(e),s.HEAPU16=new Uint16Array(e),s.HEAPU32=se=new Uint32Array(e),s.HEAPF32=new Float32Array(e),s.HEAPF64=ce=new Float64Array(e)}var ue=s.INITIAL_MEMORY||16777216;if(_)M=s.wasmMemory,L=s.buffer;else if(s.wasmMemory)M=s.wasmMemory;else if(M=new WebAssembly.Memory({initial:ue/65536,maximum:65536,shared:!0}),!(M.buffer instanceof SharedArrayBuffer))throw k(`requested a shared WebAssembly.Memory but the returned buffer is not a SharedArrayBuffer, indicating that while the browser has SharedArrayBuffer it does not have WebAssembly threads support - you may need to set a flag`),g&&k(`(on node you may need: --experimental-wasm-threads --experimental-wasm-bulk-memory and/or recent version)`),Error(`bad memory`);M&&(L=M.buffer),ue=L.byteLength,le(L);var de,z=[],fe=[],pe=[];function me(){return j}function he(){if(s.preRun)for(typeof s.preRun==`function`&&(s.preRun=[s.preRun]);s.preRun.length;)ve(s.preRun.shift());Ge(z)}function ge(){_||Ge(fe)}function _e(){if(!_){if(s.postRun)for(typeof s.postRun==`function`&&(s.postRun=[s.postRun]);s.postRun.length;)be(s.postRun.shift());Ge(pe)}}function ve(e){z.unshift(e)}function ye(e){fe.unshift(e)}function be(e){pe.unshift(e)}var xe=0,Se=null,Ce=null;function we(e){xe++,s.monitorRunDependencies&&s.monitorRunDependencies(xe)}function Te(e){if(xe--,s.monitorRunDependencies&&s.monitorRunDependencies(xe),xe==0&&(Se!==null&&(clearInterval(Se),Se=null),Ce)){var t=Ce;Ce=null,t()}}function Ee(e){s.onAbort&&s.onAbort(e),e=`Aborted(`+e+`)`,k(e),ee=!0,P=1,e+=`. Build with -sASSERTIONS for more info.`;var t=new WebAssembly.RuntimeError(e);throw l(t),t}var De=`data:application/octet-stream;base64,`;function Oe(e){return e.startsWith(De)}function ke(e){return e.startsWith(`file://`)}var Ae=`tfjs-backend-wasm-threaded-simd.wasm`;Oe(Ae)||(Ae=y(Ae));function je(e){try{if(e==Ae&&A)return new Uint8Array(A);if(S)return S(e);throw`both async and sync fetching of the wasm failed`}catch(e){Ee(e)}}function Me(){if(!A&&(m||h)){if(typeof fetch==`function`&&!ke(Ae))return fetch(Ae,{credentials:`same-origin`}).then(function(e){if(!e.ok)throw`failed to load wasm binary file at '`+Ae+`'`;return e.arrayBuffer()}).catch(function(){return je(Ae)});if(x)return new Promise(function(e,t){x(Ae,function(t){e(new Uint8Array(t))},t)})}return Promise.resolve().then(function(){return je(Ae)})}function Ne(){var e={env:Pt,wasi_snapshot_preview1:Pt};function t(e,t){if(s.asm=e.exports,Ze(s.asm._emscripten_tls_init),de=s.asm.__indirect_function_table,ye(s.asm.__wasm_call_ctors),N=t,!_){var n=B.unusedWorkers.length;B.unusedWorkers.forEach(function(e){B.loadWasmModuleToWorker(e,function(){--n||Te(`wasm-instantiate`)})})}}_||we(`wasm-instantiate`);function n(e){t(e.instance,e.module)}function r(t){return Me().then(function(t){return WebAssembly.instantiate(t,e)}).then(function(e){return e}).then(t,function(e){k(`failed to asynchronously prepare wasm: `+e),Ee(e)})}function i(){return!A&&typeof WebAssembly.instantiateStreaming==`function`&&!Oe(Ae)&&!ke(Ae)&&!g&&typeof fetch==`function`?fetch(Ae,{credentials:`same-origin`}).then(function(t){return WebAssembly.instantiateStreaming(t,e).then(n,function(e){return k(`wasm streaming compile failed: `+e),k(`falling back to ArrayBuffer instantiation`),r(n)})}):r(n)}if(s.instantiateWasm)try{return s.instantiateWasm(e,t)}catch(e){k(`Module.instantiateWasm callback failed with error: `+e),l(e)}return i().catch(l),{}}var Pe={};function Fe(e){this.name=`ExitStatus`,this.message=`Program terminated with exit(`+e+`)`,this.status=e}function Ie(e){var t=B.pthreads[e];delete B.pthreads[e],t.terminate(),Ei(e),B.runningWorkers.splice(B.runningWorkers.indexOf(t),1),t.pthread_ptr=0}function Le(e){B.pthreads[e].postMessage({cmd:`cancel`})}function Re(e){var t=B.pthreads[e];te(t),B.returnWorkerToPool(t)}function ze(e){var t=B.getNewWorker();if(!t)return 6;B.runningWorkers.push(t),B.pthreads[e.pthread_ptr]=t,t.pthread_ptr=e.pthread_ptr;var n={cmd:`run`,start_routine:e.startRoutine,arg:e.arg,pthread_ptr:e.pthread_ptr};return t.runPthread=()=>{g&&t.ref(),t.postMessage(n,e.transferList),delete t.runPthread},t.loaded&&t.runPthread(),0}var Be={varargs:void 0,get:function(){return Be.varargs+=4,i()[Be.varargs-4>>>2]},getStr:function(e){return re(e)}};function Ve(e){if(_)return vt(1,1,e);P=e,me()||(B.terminateAllThreads(),s.onExit&&s.onExit(e),ee=!0),p(e,new Fe(e))}function He(e,t){if(P=e,!t&&_)throw qe(e),`unwind`;Ve(e)}var Ue=He;function We(e){if(e instanceof Fe||e==`unwind`)return P;p(1,e)}var B={unusedWorkers:[],runningWorkers:[],tlsInitFunctions:[],pthreads:{},init:function(){_?B.initWorker():B.initMainThread()},initMainThread:function(){for(var e=8;e--;)B.allocateUnusedWorker()},initWorker:function(){j=!1},setExitStatus:function(e){P=e},terminateAllThreads:function(){for(var e of Object.values(B.pthreads))B.returnWorkerToPool(e);for(var e of B.unusedWorkers)e.terminate();B.unusedWorkers=[]},returnWorkerToPool:function(e){var t=e.pthread_ptr;delete B.pthreads[t],B.unusedWorkers.push(e),B.runningWorkers.splice(B.runningWorkers.indexOf(e),1),e.pthread_ptr=0,g&&e.unref(),Ei(t)},receiveObjectTransfer:function(e){},threadInitTLS:function(){B.tlsInitFunctions.forEach(e=>e())},loadWasmModuleToWorker:function(t,n){t.onmessage=e=>{var r=e.data,i=r.cmd;if(t.pthread_ptr&&(B.currentProxiedOperationCallerThread=t.pthread_ptr),r.targetThread&&r.targetThread!=vi()){var a=B.pthreads[r.targetThread];a?a.postMessage(r,r.transferList):k(`Internal error! Worker sent a message "`+i+`" to target pthread `+r.targetThread+`, but that thread no longer exists!`),B.currentProxiedOperationCallerThread=void 0;return}i===`processProxyingQueue`?at(r.queue):i===`spawnThread`?ze(r):i===`cleanupThread`?Re(r.thread):i===`killThread`?Ie(r.thread):i===`cancelThread`?Le(r.thread):i===`loaded`?(t.loaded=!0,g&&t.unref(),n&&n(t),t.runPthread&&t.runPthread()):i===`print`?O(`Thread `+r.threadId+`: `+r.text):i===`printErr`?k(`Thread `+r.threadId+`: `+r.text):i===`alert`?alert(`Thread `+r.threadId+`: `+r.text):r.target===`setimmediate`?t.postMessage(r):i===`callHandler`?s[r.handler](...r.args):i&&k(`worker sent an unknown command `+i),B.currentProxiedOperationCallerThread=void 0},t.onerror=e=>{throw k(`worker sent an error! `+e.filename+`:`+e.lineno+`: `+e.message),e},g&&(t.on(`message`,function(e){t.onmessage({data:e})}),t.on(`error`,function(e){t.onerror(e)}),t.on(`detachedExit`,function(){}));var r=[];for(var i of[`onExit`,`onAbort`,`print`,`printErr`])s.hasOwnProperty(i)&&r.push(i);t.postMessage({cmd:`load`,handlers:r,urlOrBlob:s.mainScriptUrlOrBlob||e,wasmMemory:M,wasmModule:N})},allocateUnusedWorker:function(){var e,t=y(`tfjs-backend-wasm-threaded-simd.worker.js`);e=new Worker(t),B.unusedWorkers.push(e)},getNewWorker:function(){return B.unusedWorkers.length==0&&(B.allocateUnusedWorker(),B.loadWasmModuleToWorker(B.unusedWorkers[0])),B.unusedWorkers.pop()}};s.PThread=B;function Ge(e){for(;e.length>0;)e.shift()(s)}function Ke(){var e=vi(),t=i()[e+52>>>2],n=t-i()[e+56>>>2];Oi(t,n),Ai(t)}s.establishStackSpace=Ke;function qe(e){if(_)return vt(2,0,e);try{Ue(e)}catch(e){We(e)}}var Je=[];function Ye(e){var t=Je[e];return t||(e>=Je.length&&(Je.length=e+1),Je[e]=t=de.get(e)),t}function Xe(e,t){var n=Ye(e)(t);me()?B.setExitStatus(n):Di(n)}s.invokeEntryPoint=Xe;function Ze(e){B.tlsInitFunctions.push(e)}function Qe(e){yi(e,!h,1,!m),B.threadInitTLS()}function $e(e){_?postMessage({cmd:`cleanupThread`,thread:e}):Re(e)}function et(e,t,n,r){return _?vt(3,1,e,t,n,r):tt(e,t,n,r)}function tt(e,t,n,r){if(typeof SharedArrayBuffer>`u`)return k(`Current environment does not support SharedArrayBuffer, pthreads are not available!`),6;var i=[];if(_&&i.length===0)return et(e,t,n,r);var a={startRoutine:n,pthread_ptr:e,arg:r,transferList:i};return _?(a.cmd=`spawnThread`,postMessage(a,i),0):ze(a)}function nt(){return 65536}var rt=!0;function it(){return rt}function at(e){Atomics.store(i(),e>>2,1),vi()&&Ti(e),Atomics.compareExchange(i(),e>>2,1,0)}s.executeNotifiedProxyingQueue=at;function ot(e,t,n,r){if(e==t)setTimeout(()=>at(r));else if(_)postMessage({targetThread:e,cmd:`processProxyingQueue`,queue:r});else{var i=B.pthreads[e];if(!i)return;i.postMessage({cmd:`processProxyingQueue`,queue:r})}return 1}function st(e,t,n){return-1}function ct(){Ee(``)}function lt(e){lt.shown||={},lt.shown[e]||(lt.shown[e]=1,g&&(e=`warning: `+e),k(e))}function ut(){g||h||lt(`Blocking on the main thread is very dangerous, see https://emscripten.org/docs/porting/pthreads.html#blocking-on-the-main-browser-thread`)}function dt(){return Date.now()}function ft(){return 4294901760}function pt(){return ft()}var mt=g?()=>{var e=process.hrtime();return e[0]*1e3+e[1]/1e6}:()=>performance.timeOrigin+performance.now();function ht(e,t,n){r().copyWithin(e>>>0,t>>>0,t+n>>>0)}function gt(){return g?oa().cpus().length:navigator.hardwareConcurrency}function _t(e){var t=ki(),n=e();return Ai(t),n}function vt(e,t){var n=arguments.length-2,r=arguments;return _t(()=>{for(var i=n,a=ji(i*8),s=a>>3,c=0;c<n;c++){var l=r[2+c];o()[s+c>>>0]=l}return Ci(e,i,a,t)})}var yt=[];function bt(e,t,n){yt.length=t;for(var r=n>>3,i=0;i<t;i++)yt[i]=o()[r+i>>>0];return(e<0?Pe[-e-1]:Nt[e]).apply(null,yt)}function xt(e){try{return M.grow(e-L.byteLength+65535>>>16),le(M.buffer),1}catch{}}function St(e){var t=r().length;if(e>>>=0,e<=t)return!1;var n=ft();if(e>n)return!1;let i=(e,t)=>e+(t-e%t)%t;for(var a=1;a<=4;a*=2){var o=t*(1+.2/a);if(o=Math.min(o,e+100663296),xt(Math.min(n,i(Math.max(e,o),65536))))return!0}return!1}function Ct(){throw`unwind`}function wt(e){return _?vt(4,1,e):52}function Tt(e,t,n,r,i){return _?vt(5,1,e,t,n,r,i):70}var Et=[null,[],[]];function Dt(e,t){var n=Et[e];t===0||t===10?((e===1?O:k)(ne(n,0)),n.length=0):n.push(t)}function Ot(e,t,n,i){if(_)return vt(6,1,e,t,n,i);for(var o=0,s=0;s<n;s++){var c=a()[t>>>2],l=a()[t+4>>>2];t+=8;for(var u=0;u<l;u++)Dt(e,r()[c+u>>>0]);o+=l}return a()[i>>>2]=o,0}function kt(e){return s[`_`+e]}function At(e,t){n().set(e,t>>>0)}function jt(e,t,n,r,i){var a={string:e=>{var t=0;if(e!=null&&e!==0){var n=(e.length<<2)+1;t=ji(n),ie(e,t,n)}return t},array:e=>{var t=ji(e.length);return At(e,t),t}};function o(e){return t===`string`?re(e):t===`boolean`?!!e:e}var s=kt(e),c=[],l=0;if(r)for(var u=0;u<r.length;u++){var d=a[n[u]];d?(l===0&&(l=ki()),c[u]=d(r[u])):c[u]=r[u]}var f=s.apply(null,c);function p(e){return l!==0&&Ai(l),o(e)}return f=p(f),f}function Mt(e,t,n,r){n||=[];var i=n.every(e=>e===`number`||e===`boolean`);return t!==`string`&&i&&!r?kt(e):function(){return jt(e,t,n,arguments,r)}}B.init();var Nt=[null,Ve,qe,et,wt,Tt,Ot],Pt={__emscripten_init_main_thread_js:Qe,__emscripten_thread_cleanup:$e,__pthread_create_js:tt,_emscripten_default_pthread_stack_size:nt,_emscripten_get_now_is_monotonic:it,_emscripten_notify_task_queue:ot,_emscripten_set_offscreencanvas_size:st,abort:ct,emscripten_check_blocking_allowed:ut,emscripten_date_now:dt,emscripten_get_heap_max:pt,emscripten_get_now:mt,emscripten_memcpy_big:ht,emscripten_num_logical_cores:gt,emscripten_receive_on_main_thread_js:bt,emscripten_resize_heap:St,emscripten_unwind_to_js_event_loop:Ct,exit:Ue,fd_close:wt,fd_seek:Tt,fd_write:Ot,memory:M||s.wasmMemory};Ne();var Ft=s.___wasm_call_ctors=function(){return(Ft=s.___wasm_call_ctors=s.asm.__wasm_call_ctors).apply(null,arguments)},It=s._init=function(){return(It=s._init=s.asm.init).apply(null,arguments)},Lt=s._init_with_threads_count=function(){return(Lt=s._init_with_threads_count=s.asm.init_with_threads_count).apply(null,arguments)},Rt=s._get_threads_count=function(){return(Rt=s._get_threads_count=s.asm.get_threads_count).apply(null,arguments)},zt=s._register_tensor=function(){return(zt=s._register_tensor=s.asm.register_tensor).apply(null,arguments)},Bt=s._dispose_data=function(){return(Bt=s._dispose_data=s.asm.dispose_data).apply(null,arguments)},Vt=s._dispose=function(){return(Vt=s._dispose=s.asm.dispose).apply(null,arguments)},Ht=s._Abs=function(){return(Ht=s._Abs=s.asm.Abs).apply(null,arguments)},Ut=s._Acos=function(){return(Ut=s._Acos=s.asm.Acos).apply(null,arguments)},Wt=s._Acosh=function(){return(Wt=s._Acosh=s.asm.Acosh).apply(null,arguments)},Gt=s._Add=function(){return(Gt=s._Add=s.asm.Add).apply(null,arguments)},Kt=s._AddN=function(){return(Kt=s._AddN=s.asm.AddN).apply(null,arguments)},qt=s._All=function(){return(qt=s._All=s.asm.All).apply(null,arguments)},Jt=s._Any=function(){return(Jt=s._Any=s.asm.Any).apply(null,arguments)},Yt=s._ArgMax=function(){return(Yt=s._ArgMax=s.asm.ArgMax).apply(null,arguments)},Xt=s._ArgMin=function(){return(Xt=s._ArgMin=s.asm.ArgMin).apply(null,arguments)},Zt=s._Asin=function(){return(Zt=s._Asin=s.asm.Asin).apply(null,arguments)},Qt=s._Asinh=function(){return(Qt=s._Asinh=s.asm.Asinh).apply(null,arguments)},$t=s._Atan=function(){return($t=s._Atan=s.asm.Atan).apply(null,arguments)},en=s._Atan2=function(){return(en=s._Atan2=s.asm.Atan2).apply(null,arguments)},tn=s._Atanh=function(){return(tn=s._Atanh=s.asm.Atanh).apply(null,arguments)},nn=s._AvgPool=function(){return(nn=s._AvgPool=s.asm.AvgPool).apply(null,arguments)},rn=s._AvgPool3D=function(){return(rn=s._AvgPool3D=s.asm.AvgPool3D).apply(null,arguments)},an=s._AvgPool3DGrad=function(){return(an=s._AvgPool3DGrad=s.asm.AvgPool3DGrad).apply(null,arguments)},on=s._AvgPoolGrad=function(){return(on=s._AvgPoolGrad=s.asm.AvgPoolGrad).apply(null,arguments)},sn=s._BatchMatMul=function(){return(sn=s._BatchMatMul=s.asm.BatchMatMul).apply(null,arguments)},cn=s._Bincount=function(){return(cn=s._Bincount=s.asm.Bincount).apply(null,arguments)},ln=s._BitwiseAnd=function(){return(ln=s._BitwiseAnd=s.asm.BitwiseAnd).apply(null,arguments)},un=s._Ceil=function(){return(un=s._Ceil=s.asm.Ceil).apply(null,arguments)},dn=s._ClipByValue=function(){return(dn=s._ClipByValue=s.asm.ClipByValue).apply(null,arguments)},fn=s._Conv2D=function(){return(fn=s._Conv2D=s.asm.Conv2D).apply(null,arguments)},pn=s._Conv2DBackpropInput=function(){return(pn=s._Conv2DBackpropInput=s.asm.Conv2DBackpropInput).apply(null,arguments)},mn=s._Conv3D=function(){return(mn=s._Conv3D=s.asm.Conv3D).apply(null,arguments)},hn=s._Conv3DBackpropFilterV2=function(){return(hn=s._Conv3DBackpropFilterV2=s.asm.Conv3DBackpropFilterV2).apply(null,arguments)},gn=s._Conv3DBackpropInputV2=function(){return(gn=s._Conv3DBackpropInputV2=s.asm.Conv3DBackpropInputV2).apply(null,arguments)},_n=s._Cos=function(){return(_n=s._Cos=s.asm.Cos).apply(null,arguments)},vn=s._Cosh=function(){return(vn=s._Cosh=s.asm.Cosh).apply(null,arguments)},yn=s._CropAndResize=function(){return(yn=s._CropAndResize=s.asm.CropAndResize).apply(null,arguments)},bn=s._Cumprod=function(){return(bn=s._Cumprod=s.asm.Cumprod).apply(null,arguments)},xn=s._Cumsum=function(){return(xn=s._Cumsum=s.asm.Cumsum).apply(null,arguments)},Sn=s._DenseBincount=function(){return(Sn=s._DenseBincount=s.asm.DenseBincount).apply(null,arguments)},Cn=s._DepthToSpace=function(){return(Cn=s._DepthToSpace=s.asm.DepthToSpace).apply(null,arguments)},wn=s._DepthwiseConv2dNative=function(){return(wn=s._DepthwiseConv2dNative=s.asm.DepthwiseConv2dNative).apply(null,arguments)},Tn=s._Diag=function(){return(Tn=s._Diag=s.asm.Diag).apply(null,arguments)},En=s._Dilation2D=function(){return(En=s._Dilation2D=s.asm.Dilation2D).apply(null,arguments)},Dn=s._Dilation2DBackpropFilter=function(){return(Dn=s._Dilation2DBackpropFilter=s.asm.Dilation2DBackpropFilter).apply(null,arguments)},On=s._Dilation2DBackpropInput=function(){return(On=s._Dilation2DBackpropInput=s.asm.Dilation2DBackpropInput).apply(null,arguments)},kn=s._Elu=function(){return(kn=s._Elu=s.asm.Elu).apply(null,arguments)},An=s._EluGrad=function(){return(An=s._EluGrad=s.asm.EluGrad).apply(null,arguments)},jn=s._Equal=function(){return(jn=s._Equal=s.asm.Equal).apply(null,arguments)},Mn=s._Erf=function(){return(Mn=s._Erf=s.asm.Erf).apply(null,arguments)},Nn=s._Exp=function(){return(Nn=s._Exp=s.asm.Exp).apply(null,arguments)},Pn=s._Expm1=function(){return(Pn=s._Expm1=s.asm.Expm1).apply(null,arguments)},Fn=s._FlipLeftRight=function(){return(Fn=s._FlipLeftRight=s.asm.FlipLeftRight).apply(null,arguments)},In=s._Floor=function(){return(In=s._Floor=s.asm.Floor).apply(null,arguments)},Ln=s._FloorDiv=function(){return(Ln=s._FloorDiv=s.asm.FloorDiv).apply(null,arguments)},Rn=s._FusedBatchNorm=function(){return(Rn=s._FusedBatchNorm=s.asm.FusedBatchNorm).apply(null,arguments)},zn=s._FusedConv2D=function(){return(zn=s._FusedConv2D=s.asm.FusedConv2D).apply(null,arguments)},Bn=s._FusedDepthwiseConv2D=function(){return(Bn=s._FusedDepthwiseConv2D=s.asm.FusedDepthwiseConv2D).apply(null,arguments)},Vn=s._Gather=function(){return(Vn=s._Gather=s.asm.Gather).apply(null,arguments)},Hn=s._GatherNd=function(){return(Hn=s._GatherNd=s.asm.GatherNd).apply(null,arguments)},Un=s._Greater=function(){return(Un=s._Greater=s.asm.Greater).apply(null,arguments)},Wn=s._GreaterEqual=function(){return(Wn=s._GreaterEqual=s.asm.GreaterEqual).apply(null,arguments)},Gn=s._IsFinite=function(){return(Gn=s._IsFinite=s.asm.IsFinite).apply(null,arguments)},Kn=s._IsInf=function(){return(Kn=s._IsInf=s.asm.IsInf).apply(null,arguments)},qn=s._IsNan=function(){return(qn=s._IsNan=s.asm.IsNan).apply(null,arguments)},Jn=s._LRN=function(){return(Jn=s._LRN=s.asm.LRN).apply(null,arguments)},Yn=s._LRNGrad=function(){return(Yn=s._LRNGrad=s.asm.LRNGrad).apply(null,arguments)},Xn=s._LeakyRelu=function(){return(Xn=s._LeakyRelu=s.asm.LeakyRelu).apply(null,arguments)},Zn=s._Less=function(){return(Zn=s._Less=s.asm.Less).apply(null,arguments)},Qn=s._LessEqual=function(){return(Qn=s._LessEqual=s.asm.LessEqual).apply(null,arguments)},$n=s._LinSpace=function(){return($n=s._LinSpace=s.asm.LinSpace).apply(null,arguments)},er=s._Log=function(){return(er=s._Log=s.asm.Log).apply(null,arguments)},tr=s._Log1p=function(){return(tr=s._Log1p=s.asm.Log1p).apply(null,arguments)},nr=s._LogicalAnd=function(){return(nr=s._LogicalAnd=s.asm.LogicalAnd).apply(null,arguments)},rr=s._LogicalNot=function(){return(rr=s._LogicalNot=s.asm.LogicalNot).apply(null,arguments)},ir=s._LogicalOr=function(){return(ir=s._LogicalOr=s.asm.LogicalOr).apply(null,arguments)},ar=s._LogicalXor=function(){return(ar=s._LogicalXor=s.asm.LogicalXor).apply(null,arguments)},or=s._Max=function(){return(or=s._Max=s.asm.Max).apply(null,arguments)},sr=s._MaxPool=function(){return(sr=s._MaxPool=s.asm.MaxPool).apply(null,arguments)},cr=s._MaxPool3D=function(){return(cr=s._MaxPool3D=s.asm.MaxPool3D).apply(null,arguments)},lr=s._MaxPool3DGrad=function(){return(lr=s._MaxPool3DGrad=s.asm.MaxPool3DGrad).apply(null,arguments)},ur=s._MaxPoolGrad=function(){return(ur=s._MaxPoolGrad=s.asm.MaxPoolGrad).apply(null,arguments)},dr=s._MaxPoolWithArgmax=function(){return(dr=s._MaxPoolWithArgmax=s.asm.MaxPoolWithArgmax).apply(null,arguments)},V=s._Maximum=function(){return(V=s._Maximum=s.asm.Maximum).apply(null,arguments)},fr=s._Mean=function(){return(fr=s._Mean=s.asm.Mean).apply(null,arguments)},pr=s._Min=function(){return(pr=s._Min=s.asm.Min).apply(null,arguments)},mr=s._Minimum=function(){return(mr=s._Minimum=s.asm.Minimum).apply(null,arguments)},hr=s._MirrorPad=function(){return(hr=s._MirrorPad=s.asm.MirrorPad).apply(null,arguments)},gr=s._Mod=function(){return(gr=s._Mod=s.asm.Mod).apply(null,arguments)},_r=s._Multinomial=function(){return(_r=s._Multinomial=s.asm.Multinomial).apply(null,arguments)},vr=s._Multiply=function(){return(vr=s._Multiply=s.asm.Multiply).apply(null,arguments)},yr=s._Neg=function(){return(yr=s._Neg=s.asm.Neg).apply(null,arguments)},br=s._NonMaxSuppressionV3=function(){return(br=s._NonMaxSuppressionV3=s.asm.NonMaxSuppressionV3).apply(null,arguments)},xr=s._NonMaxSuppressionV4=function(){return(xr=s._NonMaxSuppressionV4=s.asm.NonMaxSuppressionV4).apply(null,arguments)},Sr=s._NonMaxSuppressionV5=function(){return(Sr=s._NonMaxSuppressionV5=s.asm.NonMaxSuppressionV5).apply(null,arguments)},Cr=s._NotEqual=function(){return(Cr=s._NotEqual=s.asm.NotEqual).apply(null,arguments)},wr=s._OneHot=function(){return(wr=s._OneHot=s.asm.OneHot).apply(null,arguments)},Tr=s._PadV2=function(){return(Tr=s._PadV2=s.asm.PadV2).apply(null,arguments)},Er=s._Pow=function(){return(Er=s._Pow=s.asm.Pow).apply(null,arguments)},Dr=s._Prelu=function(){return(Dr=s._Prelu=s.asm.Prelu).apply(null,arguments)},Or=s._Prod=function(){return(Or=s._Prod=s.asm.Prod).apply(null,arguments)},kr=s._RealDiv=function(){return(kr=s._RealDiv=s.asm.RealDiv).apply(null,arguments)},Ar=s._Reciprocal=function(){return(Ar=s._Reciprocal=s.asm.Reciprocal).apply(null,arguments)},jr=s._Relu=function(){return(jr=s._Relu=s.asm.Relu).apply(null,arguments)},Mr=s._Relu6=function(){return(Mr=s._Relu6=s.asm.Relu6).apply(null,arguments)},Nr=s._ResizeBilinear=function(){return(Nr=s._ResizeBilinear=s.asm.ResizeBilinear).apply(null,arguments)},Pr=s._ResizeBilinearGrad=function(){return(Pr=s._ResizeBilinearGrad=s.asm.ResizeBilinearGrad).apply(null,arguments)},Fr=s._ResizeNearestNeighbor=function(){return(Fr=s._ResizeNearestNeighbor=s.asm.ResizeNearestNeighbor).apply(null,arguments)},Ir=s._ResizeNearestNeighborGrad=function(){return(Ir=s._ResizeNearestNeighborGrad=s.asm.ResizeNearestNeighborGrad).apply(null,arguments)},Lr=s._Reverse=function(){return(Lr=s._Reverse=s.asm.Reverse).apply(null,arguments)},Rr=s._RotateWithOffset=function(){return(Rr=s._RotateWithOffset=s.asm.RotateWithOffset).apply(null,arguments)},zr=s._Round=function(){return(zr=s._Round=s.asm.Round).apply(null,arguments)},Br=s._Rsqrt=function(){return(Br=s._Rsqrt=s.asm.Rsqrt).apply(null,arguments)},Vr=s._ScatterNd=function(){return(Vr=s._ScatterNd=s.asm.ScatterNd).apply(null,arguments)},Hr=s._SearchSorted=function(){return(Hr=s._SearchSorted=s.asm.SearchSorted).apply(null,arguments)},Ur=s._SelectV2=function(){return(Ur=s._SelectV2=s.asm.SelectV2).apply(null,arguments)},Wr=s._Selu=function(){return(Wr=s._Selu=s.asm.Selu).apply(null,arguments)},Gr=s._Sigmoid=function(){return(Gr=s._Sigmoid=s.asm.Sigmoid).apply(null,arguments)},Kr=s._Sign=function(){return(Kr=s._Sign=s.asm.Sign).apply(null,arguments)},qr=s._Sin=function(){return(qr=s._Sin=s.asm.Sin).apply(null,arguments)},Jr=s._Sinh=function(){return(Jr=s._Sinh=s.asm.Sinh).apply(null,arguments)},Yr=s._Softmax=function(){return(Yr=s._Softmax=s.asm.Softmax).apply(null,arguments)},Xr=s._Softplus=function(){return(Xr=s._Softplus=s.asm.Softplus).apply(null,arguments)},Zr=s._SparseFillEmptyRows=function(){return(Zr=s._SparseFillEmptyRows=s.asm.SparseFillEmptyRows).apply(null,arguments)},Qr=s._SparseReshape=function(){return(Qr=s._SparseReshape=s.asm.SparseReshape).apply(null,arguments)},$r=s._SparseSegmentReduction=function(){return($r=s._SparseSegmentReduction=s.asm.SparseSegmentReduction).apply(null,arguments)},ei=s._SparseToDense=function(){return(ei=s._SparseToDense=s.asm.SparseToDense).apply(null,arguments)},ti=s._Sqrt=function(){return(ti=s._Sqrt=s.asm.Sqrt).apply(null,arguments)},ni=s._Square=function(){return(ni=s._Square=s.asm.Square).apply(null,arguments)},ri=s._SquaredDifference=function(){return(ri=s._SquaredDifference=s.asm.SquaredDifference).apply(null,arguments)},ii=s._Step=function(){return(ii=s._Step=s.asm.Step).apply(null,arguments)},ai=s._StridedSlice=function(){return(ai=s._StridedSlice=s.asm.StridedSlice).apply(null,arguments)},oi=s._Sub=function(){return(oi=s._Sub=s.asm.Sub).apply(null,arguments)},si=s._Sum=function(){return(si=s._Sum=s.asm.Sum).apply(null,arguments)},ci=s._Tan=function(){return(ci=s._Tan=s.asm.Tan).apply(null,arguments)},li=s._Tanh=function(){return(li=s._Tanh=s.asm.Tanh).apply(null,arguments)},ui=s._TensorScatterUpdate=function(){return(ui=s._TensorScatterUpdate=s.asm.TensorScatterUpdate).apply(null,arguments)},di=s._Tile=function(){return(di=s._Tile=s.asm.Tile).apply(null,arguments)},fi=s._TopK=function(){return(fi=s._TopK=s.asm.TopK).apply(null,arguments)},H=s._Transform=function(){return(H=s._Transform=s.asm.Transform).apply(null,arguments)},pi=s._Transpose=function(){return(pi=s._Transpose=s.asm.Transpose).apply(null,arguments)},mi=s.__FusedMatMul=function(){return(mi=s.__FusedMatMul=s.asm._FusedMatMul).apply(null,arguments)},hi=s._malloc=function(){return(hi=s._malloc=s.asm.malloc).apply(null,arguments)},gi=s._free=function(){return(gi=s._free=s.asm.free).apply(null,arguments)},_i=s.__emscripten_tls_init=function(){return(_i=s.__emscripten_tls_init=s.asm._emscripten_tls_init).apply(null,arguments)},vi=s._pthread_self=function(){return(vi=s._pthread_self=s.asm.pthread_self).apply(null,arguments)},U=s.___errno_location=function(){return(U=s.___errno_location=s.asm.__errno_location).apply(null,arguments)},yi=s.__emscripten_thread_init=function(){return(yi=s.__emscripten_thread_init=s.asm._emscripten_thread_init).apply(null,arguments)},bi=s.__emscripten_thread_crashed=function(){return(bi=s.__emscripten_thread_crashed=s.asm._emscripten_thread_crashed).apply(null,arguments)},xi=s._emscripten_main_thread_process_queued_calls=function(){return(xi=s._emscripten_main_thread_process_queued_calls=s.asm.emscripten_main_thread_process_queued_calls).apply(null,arguments)},Si=s._emscripten_main_browser_thread_id=function(){return(Si=s._emscripten_main_browser_thread_id=s.asm.emscripten_main_browser_thread_id).apply(null,arguments)},Ci=s._emscripten_run_in_main_runtime_thread_js=function(){return(Ci=s._emscripten_run_in_main_runtime_thread_js=s.asm.emscripten_run_in_main_runtime_thread_js).apply(null,arguments)},wi=s._emscripten_dispatch_to_thread_=function(){return(wi=s._emscripten_dispatch_to_thread_=s.asm.emscripten_dispatch_to_thread_).apply(null,arguments)},Ti=s.__emscripten_proxy_execute_task_queue=function(){return(Ti=s.__emscripten_proxy_execute_task_queue=s.asm._emscripten_proxy_execute_task_queue).apply(null,arguments)},Ei=s.__emscripten_thread_free_data=function(){return(Ei=s.__emscripten_thread_free_data=s.asm._emscripten_thread_free_data).apply(null,arguments)},Di=s.__emscripten_thread_exit=function(){return(Di=s.__emscripten_thread_exit=s.asm._emscripten_thread_exit).apply(null,arguments)},Oi=s._emscripten_stack_set_limits=function(){return(Oi=s._emscripten_stack_set_limits=s.asm.emscripten_stack_set_limits).apply(null,arguments)},ki=s.stackSave=function(){return(ki=s.stackSave=s.asm.stackSave).apply(null,arguments)},Ai=s.stackRestore=function(){return(Ai=s.stackRestore=s.asm.stackRestore).apply(null,arguments)},ji=s.stackAlloc=function(){return(ji=s.stackAlloc=s.asm.stackAlloc).apply(null,arguments)},Mi=s.dynCall_iijjiiii=function(){return(Mi=s.dynCall_iijjiiii=s.asm.dynCall_iijjiiii).apply(null,arguments)},Ni=s.dynCall_jiji=function(){return(Ni=s.dynCall_jiji=s.asm.dynCall_jiji).apply(null,arguments)};s.keepRuntimeAlive=me,s.wasmMemory=M,s.cwrap=Mt,s.ExitStatus=Fe,s.PThread=B;var Pi;Ce=function e(){Pi||Fi(),Pi||(Ce=e)};function Fi(e){if(e||=f,xe>0)return;if(_){c(s),ge(),startWorker(s);return}if(he(),xe>0)return;function t(){Pi||(Pi=!0,s.calledRun=!0,!ee&&(ge(),c(s),s.onRuntimeInitialized&&s.onRuntimeInitialized(),_e()))}s.setStatus?(s.setStatus(`Running...`),setTimeout(function(){setTimeout(function(){s.setStatus(``)},1),t()},1)):t()}if(s.preInit)for(typeof s.preInit==`function`&&(s.preInit=[s.preInit]);s.preInit.length>0;)s.preInit.pop()();Fi();var Ii;u&&(Ii={uncaughtException:process.listeners(`uncaughtException`).filter(function(e){return!u.uncaughtException.indexOf(e)>-1}),unhandledRejection:process.listeners(`unhandledRejection`).filter(function(e){return!u.unhandledRejection.indexOf(e)>-1})});var Li;if(typeof WasmBackendModule<`u`)Li=WasmBackendModule;else if(t!==void 0)Li=t;else throw Error(`Could not find wasm module in post.js`);if(Ii){var Ri=Li._dispose;Li._dispose=function(){Ri(),Ii.uncaughtException.forEach(function(e){process.removeListener(`uncaughtException`,e)}),Ii.unhandledRejection.forEach(function(e){process.removeListener(`unhandledRejection`,e)})}}return t.ready})})();typeof e==`object`&&typeof t==`object`?t.exports=n:typeof define==`function`&&define.amd?define([],function(){return n}):typeof e==`object`&&(e.WasmBackendModuleThreadedSimd=n)})),uD=o(((e,t)=>{t.exports.wasmWorkerContents=`"use strict";var Module={};var ENVIRONMENT_IS_NODE=typeof process=="object"&&typeof process.versions=="object"&&typeof process.versions.node=="string";if(ENVIRONMENT_IS_NODE){var nodeWorkerThreads=require("worker_threads");var parentPort=nodeWorkerThreads.parentPort;parentPort.on("message",data=>onmessage({data:data}));var fs=require("fs");Object.assign(global,{self:global,require:require,Module:Module,location:{href:__filename},Worker:nodeWorkerThreads.Worker,importScripts:function(f){(0,eval)(fs.readFileSync(f,"utf8")+"//# sourceURL="+f)},postMessage:function(msg){parentPort.postMessage(msg)},performance:global.performance||{now:function(){return Date.now()}}})}var initializedJS=false;var pendingNotifiedProxyingQueues=[];function threadPrintErr(){var text=Array.prototype.slice.call(arguments).join(" ");if(ENVIRONMENT_IS_NODE){fs.writeSync(2,text+"
");return}console.error(text)}function threadAlert(){var text=Array.prototype.slice.call(arguments).join(" ");postMessage({cmd:"alert",text:text,threadId:Module["_pthread_self"]()})}var err=threadPrintErr;self.alert=threadAlert;Module["instantiateWasm"]=(info,receiveInstance)=>{var instance=new WebAssembly.Instance(Module["wasmModule"],info);receiveInstance(instance);Module["wasmModule"]=null;return instance.exports};self.onunhandledrejection=e=>{throw e.reason??e};self.startWorker=instance=>{Module=instance;postMessage({"cmd":"loaded"})};self.onmessage=e=>{try{if(e.data.cmd==="load"){Module["wasmModule"]=e.data.wasmModule;for(const handler of e.data.handlers){Module[handler]=function(){postMessage({cmd:"callHandler",handler:handler,args:[...arguments]})}}Module["wasmMemory"]=e.data.wasmMemory;Module["buffer"]=Module["wasmMemory"].buffer;Module["ENVIRONMENT_IS_PTHREAD"]=true;if(typeof e.data.urlOrBlob=="string"){importScripts(e.data.urlOrBlob)}else{var objectUrl=URL.createObjectURL(e.data.urlOrBlob);importScripts(objectUrl);URL.revokeObjectURL(objectUrl)}WasmBackendModuleThreadedSimd(Module)}else if(e.data.cmd==="run"){Module["__emscripten_thread_init"](e.data.pthread_ptr,0,0,1);Module["establishStackSpace"]();Module["PThread"].receiveObjectTransfer(e.data);Module["PThread"].threadInitTLS();if(!initializedJS){pendingNotifiedProxyingQueues.forEach(queue=>{Module["executeNotifiedProxyingQueue"](queue)});pendingNotifiedProxyingQueues=[];initializedJS=true}try{Module["invokeEntryPoint"](e.data.start_routine,e.data.arg)}catch(ex){if(ex!="unwind"){if(ex instanceof Module["ExitStatus"]){if(Module["keepRuntimeAlive"]()){}else{Module["__emscripten_thread_exit"](ex.status)}}else{throw ex}}}}else if(e.data.cmd==="cancel"){if(Module["_pthread_self"]()){Module["__emscripten_thread_exit"](-1)}}else if(e.data.target==="setimmediate"){}else if(e.data.cmd==="processProxyingQueue"){if(initializedJS){Module["executeNotifiedProxyingQueue"](e.data.queue)}else{pendingNotifiedProxyingQueues.push(e.data.queue)}}else if(e.data.cmd){err("worker.js received unknown command "+e.data.cmd);err(e.data)}}catch(ex){if(Module["__emscripten_thread_crashed"]){Module["__emscripten_thread_crashed"]()}throw ex}};`})),dD=o(((e,t)=>{var n=(()=>{var e=typeof document<`u`&&document.currentScript?document.currentScript.src:void 0;return typeof __filename<`u`&&(e||=__filename),(function(t){t||={};var n=t===void 0?{}:t,r,i;n.ready=new Promise(function(e,t){r=e,i=t});var a;typeof process<`u`&&process.listeners&&(a={uncaughtException:process.listeners(`uncaughtException`),unhandledRejection:process.listeners(`unhandledRejection`)});var o=Object.assign({},n),s=[],c=typeof window==`object`,l=typeof importScripts==`function`,u=typeof process==`object`&&typeof process.versions==`object`&&typeof process.versions.node==`string`,d=``;function f(e){return n.locateFile?n.locateFile(e,d):d+e}var p,m,h;if(u){var g=oa(),_=oa();d=l?_.dirname(d)+`/`:__dirname+`/`,p=(e,t)=>(e=z(e)?new URL(e):_.normalize(e),g.readFileSync(e,t?void 0:`utf8`)),h=e=>{var t=p(e,!0);return t.buffer||(t=new Uint8Array(t)),t},m=(e,t,n)=>{e=z(e)?new URL(e):_.normalize(e),g.readFile(e,function(e,r){e?n(e):t(r.buffer)})},process.argv.length>1&&process.argv[1].replace(/\\/g,`/`),s=process.argv.slice(2),process.on(`uncaughtException`,function(e){if(!(e instanceof ge))throw e}),process.on(`unhandledRejection`,function(e){throw e}),n.inspect=function(){return`[Emscripten Module object]`}}else(c||l)&&(l?d=self.location.href:typeof document<`u`&&document.currentScript&&(d=document.currentScript.src),e&&(d=e),d=d.indexOf(`blob:`)===0?``:d.substr(0,d.replace(/[?#].*/,``).lastIndexOf(`/`)+1),p=e=>{var t=new XMLHttpRequest;return t.open(`GET`,e,!1),t.send(null),t.responseText},l&&(h=e=>{var t=new XMLHttpRequest;return t.open(`GET`,e,!1),t.responseType=`arraybuffer`,t.send(null),new Uint8Array(t.response)}),m=(e,t,n)=>{var r=new XMLHttpRequest;r.open(`GET`,e,!0),r.responseType=`arraybuffer`,r.onload=()=>{if(r.status==200||r.status==0&&r.response){t(r.response);return}n()},r.onerror=n,r.send(null)});var v=n.print||console.log.bind(console),y=n.printErr||console.warn.bind(console);Object.assign(n,o),o=null,n.arguments&&(s=n.arguments),n.thisProgram&&n.thisProgram,n.quit&&n.quit;var b;n.wasmBinary&&(b=n.wasmBinary),n.noExitRuntime,typeof WebAssembly!=`object`&&le(`no native wasm support detected`);var x,S=!1,C=typeof TextDecoder<`u`?new TextDecoder(`utf8`):void 0;function w(e,t,n){t>>>=0;for(var r=t+n,i=t;e[i]&&!(i>=r);)++i;if(i-t>16&&e.buffer&&C)return C.decode(e.subarray(t,i));for(var a=``;t<i;){var o=e[t++];if(!(o&128)){a+=String.fromCharCode(o);continue}var s=e[t++]&63;if((o&224)==192){a+=String.fromCharCode((o&31)<<6|s);continue}var c=e[t++]&63;if(o=(o&240)==224?(o&15)<<12|s<<6|c:(o&7)<<18|s<<12|c<<6|e[t++]&63,o<65536)a+=String.fromCharCode(o);else{var l=o-65536;a+=String.fromCharCode(55296|l>>10,56320|l&1023)}}return a}function T(e,t){return e>>>=0,e?w(A,e,t):``}function E(e,t,n,r){if(n>>>=0,!(r>0))return 0;for(var i=n,a=n+r-1,o=0;o<e.length;++o){var s=e.charCodeAt(o);if(s>=55296&&s<=57343){var c=e.charCodeAt(++o);s=65536+((s&1023)<<10)|c&1023}if(s<=127){if(n>=a)break;t[n++>>>0]=s}else if(s<=2047){if(n+1>=a)break;t[n++>>>0]=192|s>>6,t[n++>>>0]=128|s&63}else if(s<=65535){if(n+2>=a)break;t[n++>>>0]=224|s>>12,t[n++>>>0]=128|s>>6&63,t[n++>>>0]=128|s&63}else{if(n+3>=a)break;t[n++>>>0]=240|s>>18,t[n++>>>0]=128|s>>12&63,t[n++>>>0]=128|s>>6&63,t[n++>>>0]=128|s&63}}return t[n>>>0]=0,n-i}function D(e,t,n){return E(e,A,t,n)}var O,k,A,j,M;function N(e){O=e,n.HEAP8=k=new Int8Array(e),n.HEAP16=new Int16Array(e),n.HEAP32=j=new Int32Array(e),n.HEAPU8=A=new Uint8Array(e),n.HEAPU16=new Uint16Array(e),n.HEAPU32=M=new Uint32Array(e),n.HEAPF32=new Float32Array(e),n.HEAPF64=new Float64Array(e)}n.INITIAL_MEMORY;var ee=[],P=[],te=[];function F(){if(n.preRun)for(typeof n.preRun==`function`&&(n.preRun=[n.preRun]);n.preRun.length;)I(n.preRun.shift());_e(ee)}function ne(){_e(P)}function re(){if(n.postRun)for(typeof n.postRun==`function`&&(n.postRun=[n.postRun]);n.postRun.length;)L(n.postRun.shift());_e(te)}function I(e){ee.unshift(e)}function ie(e){P.unshift(e)}function L(e){te.unshift(e)}var R=0,ae=null,oe=null;function se(e){R++,n.monitorRunDependencies&&n.monitorRunDependencies(R)}function ce(e){if(R--,n.monitorRunDependencies&&n.monitorRunDependencies(R),R==0&&(ae!==null&&(clearInterval(ae),ae=null),oe)){var t=oe;oe=null,t()}}function le(e){n.onAbort&&n.onAbort(e),e=`Aborted(`+e+`)`,y(e),S=!0,e+=`. Build with -sASSERTIONS for more info.`;var t=new WebAssembly.RuntimeError(e);throw i(t),t}var ue=`data:application/octet-stream;base64,`;function de(e){return e.startsWith(ue)}function z(e){return e.startsWith(`file://`)}var fe=`tfjs-backend-wasm.wasm`;de(fe)||(fe=f(fe));function pe(e){try{if(e==fe&&b)return new Uint8Array(b);if(h)return h(e);throw`both async and sync fetching of the wasm failed`}catch(e){le(e)}}function me(){if(!b&&(c||l)){if(typeof fetch==`function`&&!z(fe))return fetch(fe,{credentials:`same-origin`}).then(function(e){if(!e.ok)throw`failed to load wasm binary file at '`+fe+`'`;return e.arrayBuffer()}).catch(function(){return pe(fe)});if(m)return new Promise(function(e,t){m(fe,function(t){e(new Uint8Array(t))},t)})}return Promise.resolve().then(function(){return pe(fe)})}function he(){var e={env:Pe,wasi_snapshot_preview1:Pe};function t(e,t){n.asm=e.exports,x=n.asm.memory,N(x.buffer),n.asm.__indirect_function_table,ie(n.asm.__wasm_call_ctors),ce(`wasm-instantiate`)}se(`wasm-instantiate`);function r(e){t(e.instance)}function a(t){return me().then(function(t){return WebAssembly.instantiate(t,e)}).then(function(e){return e}).then(t,function(e){y(`failed to asynchronously prepare wasm: `+e),le(e)})}function o(){return!b&&typeof WebAssembly.instantiateStreaming==`function`&&!de(fe)&&!z(fe)&&!u&&typeof fetch==`function`?fetch(fe,{credentials:`same-origin`}).then(function(t){return WebAssembly.instantiateStreaming(t,e).then(r,function(e){return y(`wasm streaming compile failed: `+e),y(`falling back to ArrayBuffer instantiation`),a(r)})}):a(r)}if(n.instantiateWasm)try{return n.instantiateWasm(e,t)}catch(e){y(`Module.instantiateWasm callback failed with error: `+e),i(e)}return o().catch(i),{}}function ge(e){this.name=`ExitStatus`,this.message=`Program terminated with exit(`+e+`)`,this.status=e}function _e(e){for(;e.length>0;)e.shift()(n)}function ve(){le(``)}function ye(){return 4294901760}function be(){return ye()}function xe(e,t,n){A.copyWithin(e>>>0,t>>>0,t+n>>>0)}function Se(e){try{return x.grow(e-O.byteLength+65535>>>16),N(x.buffer),1}catch{}}function Ce(e){var t=A.length;e>>>=0;var n=ye();if(e>n)return!1;let r=(e,t)=>e+(t-e%t)%t;for(var i=1;i<=4;i*=2){var a=t*(1+.2/i);if(a=Math.min(a,e+100663296),Se(Math.min(n,r(Math.max(e,a),65536))))return!0}return!1}var we={varargs:void 0,get:function(){return we.varargs+=4,j[we.varargs-4>>>2]},getStr:function(e){return T(e)}};function Te(e){return 52}function Ee(e,t,n,r,i){return 70}var De=[null,[],[]];function Oe(e,t){var n=De[e];t===0||t===10?((e===1?v:y)(w(n,0)),n.length=0):n.push(t)}function ke(e,t,n,r){for(var i=0,a=0;a<n;a++){var o=M[t>>>2],s=M[t+4>>>2];t+=8;for(var c=0;c<s;c++)Oe(e,A[o+c>>>0]);i+=s}return M[r>>>2]=i,0}function Ae(e){return n[`_`+e]}function je(e,t){k.set(e,t>>>0)}function Me(e,t,n,r,i){var a={string:e=>{var t=0;if(e!=null&&e!==0){var n=(e.length<<2)+1;t=br(n),D(e,t,n)}return t},array:e=>{var t=br(e.length);return je(e,t),t}};function o(e){return t===`string`?T(e):t===`boolean`?!!e:e}var s=Ae(e),c=[],l=0;if(r)for(var u=0;u<r.length;u++){var d=a[n[u]];d?(l===0&&(l=vr()),c[u]=d(r[u])):c[u]=r[u]}var f=s.apply(null,c);function p(e){return l!==0&&yr(l),o(e)}return f=p(f),f}function Ne(e,t,n,r){n||=[];var i=n.every(e=>e===`number`||e===`boolean`);return t!==`string`&&i&&!r?Ae(e):function(){return Me(e,t,n,arguments,r)}}var Pe={abort:ve,emscripten_get_heap_max:be,emscripten_memcpy_big:xe,emscripten_resize_heap:Ce,fd_close:Te,fd_seek:Ee,fd_write:ke};he();var Fe=n.___wasm_call_ctors=function(){return(Fe=n.___wasm_call_ctors=n.asm.__wasm_call_ctors).apply(null,arguments)},Ie=n._init=function(){return(Ie=n._init=n.asm.init).apply(null,arguments)},Le=n._init_with_threads_count=function(){return(Le=n._init_with_threads_count=n.asm.init_with_threads_count).apply(null,arguments)},Re=n._get_threads_count=function(){return(Re=n._get_threads_count=n.asm.get_threads_count).apply(null,arguments)},ze=n._register_tensor=function(){return(ze=n._register_tensor=n.asm.register_tensor).apply(null,arguments)},Be=n._dispose_data=function(){return(Be=n._dispose_data=n.asm.dispose_data).apply(null,arguments)},Ve=n._dispose=function(){return(Ve=n._dispose=n.asm.dispose).apply(null,arguments)},He=n._Abs=function(){return(He=n._Abs=n.asm.Abs).apply(null,arguments)},Ue=n._Acos=function(){return(Ue=n._Acos=n.asm.Acos).apply(null,arguments)},We=n._Acosh=function(){return(We=n._Acosh=n.asm.Acosh).apply(null,arguments)},B=n._Add=function(){return(B=n._Add=n.asm.Add).apply(null,arguments)},Ge=n._AddN=function(){return(Ge=n._AddN=n.asm.AddN).apply(null,arguments)},Ke=n._All=function(){return(Ke=n._All=n.asm.All).apply(null,arguments)},qe=n._Any=function(){return(qe=n._Any=n.asm.Any).apply(null,arguments)},Je=n._ArgMax=function(){return(Je=n._ArgMax=n.asm.ArgMax).apply(null,arguments)},Ye=n._ArgMin=function(){return(Ye=n._ArgMin=n.asm.ArgMin).apply(null,arguments)},Xe=n._Asin=function(){return(Xe=n._Asin=n.asm.Asin).apply(null,arguments)},Ze=n._Asinh=function(){return(Ze=n._Asinh=n.asm.Asinh).apply(null,arguments)},Qe=n._Atan=function(){return(Qe=n._Atan=n.asm.Atan).apply(null,arguments)},$e=n._Atan2=function(){return($e=n._Atan2=n.asm.Atan2).apply(null,arguments)},et=n._Atanh=function(){return(et=n._Atanh=n.asm.Atanh).apply(null,arguments)},tt=n._AvgPool=function(){return(tt=n._AvgPool=n.asm.AvgPool).apply(null,arguments)},nt=n._AvgPool3D=function(){return(nt=n._AvgPool3D=n.asm.AvgPool3D).apply(null,arguments)},rt=n._AvgPool3DGrad=function(){return(rt=n._AvgPool3DGrad=n.asm.AvgPool3DGrad).apply(null,arguments)},it=n._AvgPoolGrad=function(){return(it=n._AvgPoolGrad=n.asm.AvgPoolGrad).apply(null,arguments)},at=n._BatchMatMul=function(){return(at=n._BatchMatMul=n.asm.BatchMatMul).apply(null,arguments)},ot=n._Bincount=function(){return(ot=n._Bincount=n.asm.Bincount).apply(null,arguments)},st=n._BitwiseAnd=function(){return(st=n._BitwiseAnd=n.asm.BitwiseAnd).apply(null,arguments)},ct=n._Ceil=function(){return(ct=n._Ceil=n.asm.Ceil).apply(null,arguments)},lt=n._ClipByValue=function(){return(lt=n._ClipByValue=n.asm.ClipByValue).apply(null,arguments)},ut=n._Conv2D=function(){return(ut=n._Conv2D=n.asm.Conv2D).apply(null,arguments)},dt=n._Conv2DBackpropInput=function(){return(dt=n._Conv2DBackpropInput=n.asm.Conv2DBackpropInput).apply(null,arguments)},ft=n._Conv3D=function(){return(ft=n._Conv3D=n.asm.Conv3D).apply(null,arguments)},pt=n._Conv3DBackpropFilterV2=function(){return(pt=n._Conv3DBackpropFilterV2=n.asm.Conv3DBackpropFilterV2).apply(null,arguments)},mt=n._Conv3DBackpropInputV2=function(){return(mt=n._Conv3DBackpropInputV2=n.asm.Conv3DBackpropInputV2).apply(null,arguments)},ht=n._Cos=function(){return(ht=n._Cos=n.asm.Cos).apply(null,arguments)},gt=n._Cosh=function(){return(gt=n._Cosh=n.asm.Cosh).apply(null,arguments)},_t=n._CropAndResize=function(){return(_t=n._CropAndResize=n.asm.CropAndResize).apply(null,arguments)},vt=n._Cumprod=function(){return(vt=n._Cumprod=n.asm.Cumprod).apply(null,arguments)},yt=n._Cumsum=function(){return(yt=n._Cumsum=n.asm.Cumsum).apply(null,arguments)},bt=n._DenseBincount=function(){return(bt=n._DenseBincount=n.asm.DenseBincount).apply(null,arguments)},xt=n._DepthToSpace=function(){return(xt=n._DepthToSpace=n.asm.DepthToSpace).apply(null,arguments)},St=n._DepthwiseConv2dNative=function(){return(St=n._DepthwiseConv2dNative=n.asm.DepthwiseConv2dNative).apply(null,arguments)},Ct=n._Diag=function(){return(Ct=n._Diag=n.asm.Diag).apply(null,arguments)},wt=n._Dilation2D=function(){return(wt=n._Dilation2D=n.asm.Dilation2D).apply(null,arguments)},Tt=n._Dilation2DBackpropFilter=function(){return(Tt=n._Dilation2DBackpropFilter=n.asm.Dilation2DBackpropFilter).apply(null,arguments)},Et=n._Dilation2DBackpropInput=function(){return(Et=n._Dilation2DBackpropInput=n.asm.Dilation2DBackpropInput).apply(null,arguments)},Dt=n._Elu=function(){return(Dt=n._Elu=n.asm.Elu).apply(null,arguments)},Ot=n._EluGrad=function(){return(Ot=n._EluGrad=n.asm.EluGrad).apply(null,arguments)},kt=n._Equal=function(){return(kt=n._Equal=n.asm.Equal).apply(null,arguments)},At=n._Erf=function(){return(At=n._Erf=n.asm.Erf).apply(null,arguments)},jt=n._Exp=function(){return(jt=n._Exp=n.asm.Exp).apply(null,arguments)},Mt=n._Expm1=function(){return(Mt=n._Expm1=n.asm.Expm1).apply(null,arguments)},Nt=n._FlipLeftRight=function(){return(Nt=n._FlipLeftRight=n.asm.FlipLeftRight).apply(null,arguments)},Pt=n._Floor=function(){return(Pt=n._Floor=n.asm.Floor).apply(null,arguments)},Ft=n._FloorDiv=function(){return(Ft=n._FloorDiv=n.asm.FloorDiv).apply(null,arguments)},It=n._FusedBatchNorm=function(){return(It=n._FusedBatchNorm=n.asm.FusedBatchNorm).apply(null,arguments)},Lt=n._FusedConv2D=function(){return(Lt=n._FusedConv2D=n.asm.FusedConv2D).apply(null,arguments)},Rt=n._FusedDepthwiseConv2D=function(){return(Rt=n._FusedDepthwiseConv2D=n.asm.FusedDepthwiseConv2D).apply(null,arguments)},zt=n._Gather=function(){return(zt=n._Gather=n.asm.Gather).apply(null,arguments)},Bt=n._GatherNd=function(){return(Bt=n._GatherNd=n.asm.GatherNd).apply(null,arguments)},Vt=n._Greater=function(){return(Vt=n._Greater=n.asm.Greater).apply(null,arguments)},Ht=n._GreaterEqual=function(){return(Ht=n._GreaterEqual=n.asm.GreaterEqual).apply(null,arguments)},Ut=n._IsFinite=function(){return(Ut=n._IsFinite=n.asm.IsFinite).apply(null,arguments)},Wt=n._IsInf=function(){return(Wt=n._IsInf=n.asm.IsInf).apply(null,arguments)},Gt=n._IsNan=function(){return(Gt=n._IsNan=n.asm.IsNan).apply(null,arguments)},Kt=n._LRN=function(){return(Kt=n._LRN=n.asm.LRN).apply(null,arguments)},qt=n._LRNGrad=function(){return(qt=n._LRNGrad=n.asm.LRNGrad).apply(null,arguments)},Jt=n._LeakyRelu=function(){return(Jt=n._LeakyRelu=n.asm.LeakyRelu).apply(null,arguments)},Yt=n._Less=function(){return(Yt=n._Less=n.asm.Less).apply(null,arguments)},Xt=n._LessEqual=function(){return(Xt=n._LessEqual=n.asm.LessEqual).apply(null,arguments)},Zt=n._LinSpace=function(){return(Zt=n._LinSpace=n.asm.LinSpace).apply(null,arguments)},Qt=n._Log=function(){return(Qt=n._Log=n.asm.Log).apply(null,arguments)},$t=n._Log1p=function(){return($t=n._Log1p=n.asm.Log1p).apply(null,arguments)},en=n._LogicalAnd=function(){return(en=n._LogicalAnd=n.asm.LogicalAnd).apply(null,arguments)},tn=n._LogicalNot=function(){return(tn=n._LogicalNot=n.asm.LogicalNot).apply(null,arguments)},nn=n._LogicalOr=function(){return(nn=n._LogicalOr=n.asm.LogicalOr).apply(null,arguments)},rn=n._LogicalXor=function(){return(rn=n._LogicalXor=n.asm.LogicalXor).apply(null,arguments)},an=n._Max=function(){return(an=n._Max=n.asm.Max).apply(null,arguments)},on=n._MaxPool=function(){return(on=n._MaxPool=n.asm.MaxPool).apply(null,arguments)},sn=n._MaxPool3D=function(){return(sn=n._MaxPool3D=n.asm.MaxPool3D).apply(null,arguments)},cn=n._MaxPool3DGrad=function(){return(cn=n._MaxPool3DGrad=n.asm.MaxPool3DGrad).apply(null,arguments)},ln=n._MaxPoolGrad=function(){return(ln=n._MaxPoolGrad=n.asm.MaxPoolGrad).apply(null,arguments)},un=n._MaxPoolWithArgmax=function(){return(un=n._MaxPoolWithArgmax=n.asm.MaxPoolWithArgmax).apply(null,arguments)},dn=n._Maximum=function(){return(dn=n._Maximum=n.asm.Maximum).apply(null,arguments)},fn=n._Mean=function(){return(fn=n._Mean=n.asm.Mean).apply(null,arguments)},pn=n._Min=function(){return(pn=n._Min=n.asm.Min).apply(null,arguments)},mn=n._Minimum=function(){return(mn=n._Minimum=n.asm.Minimum).apply(null,arguments)},hn=n._MirrorPad=function(){return(hn=n._MirrorPad=n.asm.MirrorPad).apply(null,arguments)},gn=n._Mod=function(){return(gn=n._Mod=n.asm.Mod).apply(null,arguments)},_n=n._Multinomial=function(){return(_n=n._Multinomial=n.asm.Multinomial).apply(null,arguments)},vn=n._Multiply=function(){return(vn=n._Multiply=n.asm.Multiply).apply(null,arguments)},yn=n._Neg=function(){return(yn=n._Neg=n.asm.Neg).apply(null,arguments)},bn=n._NonMaxSuppressionV3=function(){return(bn=n._NonMaxSuppressionV3=n.asm.NonMaxSuppressionV3).apply(null,arguments)},xn=n._NonMaxSuppressionV4=function(){return(xn=n._NonMaxSuppressionV4=n.asm.NonMaxSuppressionV4).apply(null,arguments)},Sn=n._NonMaxSuppressionV5=function(){return(Sn=n._NonMaxSuppressionV5=n.asm.NonMaxSuppressionV5).apply(null,arguments)},Cn=n._NotEqual=function(){return(Cn=n._NotEqual=n.asm.NotEqual).apply(null,arguments)},wn=n._OneHot=function(){return(wn=n._OneHot=n.asm.OneHot).apply(null,arguments)},Tn=n._PadV2=function(){return(Tn=n._PadV2=n.asm.PadV2).apply(null,arguments)},En=n._Pow=function(){return(En=n._Pow=n.asm.Pow).apply(null,arguments)},Dn=n._Prelu=function(){return(Dn=n._Prelu=n.asm.Prelu).apply(null,arguments)},On=n._Prod=function(){return(On=n._Prod=n.asm.Prod).apply(null,arguments)},kn=n._RealDiv=function(){return(kn=n._RealDiv=n.asm.RealDiv).apply(null,arguments)},An=n._Reciprocal=function(){return(An=n._Reciprocal=n.asm.Reciprocal).apply(null,arguments)},jn=n._Relu=function(){return(jn=n._Relu=n.asm.Relu).apply(null,arguments)},Mn=n._Relu6=function(){return(Mn=n._Relu6=n.asm.Relu6).apply(null,arguments)},Nn=n._ResizeBilinear=function(){return(Nn=n._ResizeBilinear=n.asm.ResizeBilinear).apply(null,arguments)},Pn=n._ResizeBilinearGrad=function(){return(Pn=n._ResizeBilinearGrad=n.asm.ResizeBilinearGrad).apply(null,arguments)},Fn=n._ResizeNearestNeighbor=function(){return(Fn=n._ResizeNearestNeighbor=n.asm.ResizeNearestNeighbor).apply(null,arguments)},In=n._ResizeNearestNeighborGrad=function(){return(In=n._ResizeNearestNeighborGrad=n.asm.ResizeNearestNeighborGrad).apply(null,arguments)},Ln=n._Reverse=function(){return(Ln=n._Reverse=n.asm.Reverse).apply(null,arguments)},Rn=n._RotateWithOffset=function(){return(Rn=n._RotateWithOffset=n.asm.RotateWithOffset).apply(null,arguments)},zn=n._Round=function(){return(zn=n._Round=n.asm.Round).apply(null,arguments)},Bn=n._Rsqrt=function(){return(Bn=n._Rsqrt=n.asm.Rsqrt).apply(null,arguments)},Vn=n._ScatterNd=function(){return(Vn=n._ScatterNd=n.asm.ScatterNd).apply(null,arguments)},Hn=n._SearchSorted=function(){return(Hn=n._SearchSorted=n.asm.SearchSorted).apply(null,arguments)},Un=n._SelectV2=function(){return(Un=n._SelectV2=n.asm.SelectV2).apply(null,arguments)},Wn=n._Selu=function(){return(Wn=n._Selu=n.asm.Selu).apply(null,arguments)},Gn=n._Sigmoid=function(){return(Gn=n._Sigmoid=n.asm.Sigmoid).apply(null,arguments)},Kn=n._Sign=function(){return(Kn=n._Sign=n.asm.Sign).apply(null,arguments)},qn=n._Sin=function(){return(qn=n._Sin=n.asm.Sin).apply(null,arguments)},Jn=n._Sinh=function(){return(Jn=n._Sinh=n.asm.Sinh).apply(null,arguments)},Yn=n._Softmax=function(){return(Yn=n._Softmax=n.asm.Softmax).apply(null,arguments)},Xn=n._Softplus=function(){return(Xn=n._Softplus=n.asm.Softplus).apply(null,arguments)},Zn=n._SparseFillEmptyRows=function(){return(Zn=n._SparseFillEmptyRows=n.asm.SparseFillEmptyRows).apply(null,arguments)},Qn=n._SparseReshape=function(){return(Qn=n._SparseReshape=n.asm.SparseReshape).apply(null,arguments)},$n=n._SparseSegmentReduction=function(){return($n=n._SparseSegmentReduction=n.asm.SparseSegmentReduction).apply(null,arguments)},er=n._SparseToDense=function(){return(er=n._SparseToDense=n.asm.SparseToDense).apply(null,arguments)},tr=n._Sqrt=function(){return(tr=n._Sqrt=n.asm.Sqrt).apply(null,arguments)},nr=n._Square=function(){return(nr=n._Square=n.asm.Square).apply(null,arguments)},rr=n._SquaredDifference=function(){return(rr=n._SquaredDifference=n.asm.SquaredDifference).apply(null,arguments)},ir=n._Step=function(){return(ir=n._Step=n.asm.Step).apply(null,arguments)},ar=n._StridedSlice=function(){return(ar=n._StridedSlice=n.asm.StridedSlice).apply(null,arguments)},or=n._Sub=function(){return(or=n._Sub=n.asm.Sub).apply(null,arguments)},sr=n._Sum=function(){return(sr=n._Sum=n.asm.Sum).apply(null,arguments)},cr=n._Tan=function(){return(cr=n._Tan=n.asm.Tan).apply(null,arguments)},lr=n._Tanh=function(){return(lr=n._Tanh=n.asm.Tanh).apply(null,arguments)},ur=n._TensorScatterUpdate=function(){return(ur=n._TensorScatterUpdate=n.asm.TensorScatterUpdate).apply(null,arguments)},dr=n._Tile=function(){return(dr=n._Tile=n.asm.Tile).apply(null,arguments)},V=n._TopK=function(){return(V=n._TopK=n.asm.TopK).apply(null,arguments)},fr=n._Transform=function(){return(fr=n._Transform=n.asm.Transform).apply(null,arguments)},pr=n._Transpose=function(){return(pr=n._Transpose=n.asm.Transpose).apply(null,arguments)},mr=n.__FusedMatMul=function(){return(mr=n.__FusedMatMul=n.asm._FusedMatMul).apply(null,arguments)},hr=n._malloc=function(){return(hr=n._malloc=n.asm.malloc).apply(null,arguments)},gr=n._free=function(){return(gr=n._free=n.asm.free).apply(null,arguments)},_r=n.___errno_location=function(){return(_r=n.___errno_location=n.asm.__errno_location).apply(null,arguments)},vr=n.stackSave=function(){return(vr=n.stackSave=n.asm.stackSave).apply(null,arguments)},yr=n.stackRestore=function(){return(yr=n.stackRestore=n.asm.stackRestore).apply(null,arguments)},br=n.stackAlloc=function(){return(br=n.stackAlloc=n.asm.stackAlloc).apply(null,arguments)},xr=n.dynCall_iijjiiii=function(){return(xr=n.dynCall_iijjiiii=n.asm.dynCall_iijjiiii).apply(null,arguments)},Sr=n.dynCall_jiji=function(){return(Sr=n.dynCall_jiji=n.asm.dynCall_jiji).apply(null,arguments)};n.cwrap=Ne;var Cr;oe=function e(){Cr||wr(),Cr||(oe=e)};function wr(e){if(e||=s,R>0||(F(),R>0))return;function t(){Cr||(Cr=!0,n.calledRun=!0,!S&&(ne(),r(n),n.onRuntimeInitialized&&n.onRuntimeInitialized(),re()))}n.setStatus?(n.setStatus(`Running...`),setTimeout(function(){setTimeout(function(){n.setStatus(``)},1),t()},1)):t()}if(n.preInit)for(typeof n.preInit==`function`&&(n.preInit=[n.preInit]);n.preInit.length>0;)n.preInit.pop()();wr();var Tr;a&&(Tr={uncaughtException:process.listeners(`uncaughtException`).filter(function(e){return!a.uncaughtException.indexOf(e)>-1}),unhandledRejection:process.listeners(`unhandledRejection`).filter(function(e){return!a.unhandledRejection.indexOf(e)>-1})});var Er;if(t!==void 0)Er=t;else if(typeof WasmBackendModuleThreadedSimd<`u`)Er=WasmBackendModuleThreadedSimd;else throw Error(`Could not find wasm module in post.js`);if(Tr){var Dr=Er._dispose;Er._dispose=function(){Dr(),Tr.uncaughtException.forEach(function(e){process.removeListener(`uncaughtException`,e)}),Tr.unhandledRejection.forEach(function(e){process.removeListener(`unhandledRejection`,e)})}}return t.ready})})();typeof e==`object`&&typeof t==`object`?t.exports=n:typeof define==`function`&&define.amd?define([],function(){return n}):typeof e==`object`&&(e.WasmBackendModule=n)})),fD=c(lD()),pD=uD(),mD=c(dD());
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
let hD=fD.default||fD,gD=mD.default||mD;var _D=class extends u{constructor(e){super(),this.wasm=e,this.dataIdNextNumber=1,this.wasm.tfjs.initWithThreadsCount(kD),AD=this.wasm.tfjs.getThreadsCount(),this.dataIdMap=new l(this,Ti())}write(e,t,n){let r={id:this.dataIdNextNumber++};return this.move(r,e,t,n,1),r}numDataIds(){return this.dataIdMap.numDataIds()}async time(e){let t=wr();return e(),{kernelMs:wr()-t}}move(e,t,n,r,i){let a=this.dataIdNextNumber++;if(r===`string`){let o=t;this.dataIdMap.set(e,{id:a,stringBytes:o,shape:n,dtype:r,memoryOffset:null,refCount:i});return}let o=_(n),s=o*O(r),c=this.wasm._malloc(s)>>>0;this.dataIdMap.set(e,{id:a,memoryOffset:c,shape:n,dtype:r,refCount:i}),this.wasm.tfjs.registerTensor(a,o,c),t!=null&&this.wasm.HEAPU8.set(new Uint8Array(t.buffer,t.byteOffset,s),c)}async read(e){return this.readSync(e)}readSync(e,t,n){let{memoryOffset:r,dtype:i,shape:a,stringBytes:o}=this.dataIdMap.get(e);if(i===`string`)return(t==null||t===0)&&(n==null||n>=o.length)?o:o.slice(t,n);t||=0,n||=_(a);let s=O(i);return xD(this.wasm.HEAPU8.slice(r+t*s,r+n*s).buffer,i)}disposeData(e,t=!1){if(this.dataIdMap.has(e)){let n=this.dataIdMap.get(e);if(n.refCount--,!t&&n.refCount>0)return!1;this.wasm._free(n.memoryOffset),this.wasm.tfjs.disposeData(n.id),this.dataIdMap.delete(e)}return!0}refCount(e){return this.dataIdMap.has(e)?this.dataIdMap.get(e).refCount:0}incRef(e){let t=this.dataIdMap.get(e);t!=null&&t.refCount++}floatPrecision(){return 32}getMemoryOffset(e){return this.dataIdMap.get(e).memoryOffset}dispose(){this.wasm.tfjs.dispose(),`PThread`in this.wasm&&this.wasm.PThread.terminateAllThreads(),this.wasm=null}memory(){return{unreliable:!1}}makeOutput(e,t,n,r){let i;if(n==null)i=this.write(r??null,e,t);else{let r=this.dataIdNextNumber++;i={id:r},this.dataIdMap.set(i,{id:r,memoryOffset:n,shape:e,dtype:t,refCount:1});let a=_(e);this.wasm.tfjs.registerTensor(r,a,n)}return{dataId:i,shape:e,dtype:t}}typedArrayFromHeap({shape:e,dtype:t,dataId:n}){let r=this.wasm.HEAPU8.buffer,{memoryOffset:i}=this.dataIdMap.get(n),a=_(e);switch(t){case`float32`:return new Float32Array(r,i,a);case`int32`:return new Int32Array(r,i,a);case`bool`:return new Uint8Array(r,i,a);default:throw Error(`Unknown dtype ${t}`)}}};function vD(e){return(t,n)=>(Tr(e,{credentials:`same-origin`}).then(r=>{r.ok||t.env.a(`failed to load wasm binary file at '${e}'`),r.arrayBuffer().then(e=>{WebAssembly.instantiate(e,t).then(e=>{n(e.instance,e.module)})})}),{})}function yD(e,t,n){if(CD!=null)return CD;let r=`tfjs-backend-wasm.wasm`;return e&&t?r=`tfjs-backend-wasm-threaded-simd.wasm`:e&&(r=`tfjs-backend-wasm-simd.wasm`),TD!=null&&TD[r]!=null?TD[r]:n+r}async function bD(){let[e,t]=await Promise.all([z().getAsync(`WASM_HAS_SIMD_SUPPORT`),z().getAsync(`WASM_HAS_MULTITHREAD_SUPPORT`)]);return new Promise((n,r)=>{let i={};i.locateFile=(n,r)=>{if(n.endsWith(`.worker.js`)){let e=pD.wasmWorkerContents.replace(/\n/g,`\\n`),t=new Blob([e],{type:`application/javascript`});return URL.createObjectURL(t)}return n.endsWith(`.wasm`)?yD(e,t,wD??r):r+n},DD&&(i.instantiateWasm=vD(yD(e,t,wD??``)));let a=!1;i.onAbort=()=>{a||ED||(ED=!0,r({message:"Make sure the server can serve the `.wasm` file relative to the bundled js file. For more details see https://github.com/tensorflow/tfjs/blob/master/tfjs-backend-wasm/README.md#using-bundlers"}))};let o;t&&e&&CD==null?(i.mainScriptUrlOrBlob=new Blob([`var WasmBackendModuleThreadedSimd = `+hD.toString()],{type:`text/javascript`}),o=hD(i)):o=gD(i),o.then(e=>{a=!0,ED=!1,e.tfjs={init:e.cwrap(`init`,null,[]),initWithThreadsCount:e.cwrap(`init_with_threads_count`,null,[`number`]),getThreadsCount:e.cwrap(`get_threads_count`,`number`,[]),registerTensor:e.cwrap(`register_tensor`,null,[`number`,`number`,`number`]),disposeData:e.cwrap(`dispose_data`,null,[`number`]),dispose:e.cwrap(`dispose`,null,[])},n({wasm:e})}).catch(r)})}function xD(e,t){switch(t){case`float32`:return new Float32Array(e);case`int32`:return new Int32Array(e);case`bool`:return new Uint8Array(e);default:throw Error(`Unknown dtype ${t}`)}}let SD=[`tfjs-backend-wasm.wasm`,`tfjs-backend-wasm-simd.wasm`,`tfjs-backend-wasm-threaded-simd.wasm`],CD=null,wD=null,TD={},ED=!1,DD=!1;function OD(e,t=!1){if(ED)throw Error("The WASM backend was already initialized. Make sure you call `setWasmPaths()` before you call `tf.setBackend()` or `tf.ready()`");if(typeof e==`string`)wD=e;else{TD=e;let t=SD.filter(e=>TD[e]==null);if(t.length>0)throw Error(`There were no entries found for the following binaries: ${t.join(`,`)}. Please either call setWasmPaths with a map providing a path for each binary, or with a string indicating the directory where all the binaries can be found.`)}DD=t}let kD=-1,AD=-1;function jD(e){kD=e}
/**
* @license
* Copyright 2020 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
Mi(`wasm`,async()=>{let{wasm:e}=await bD();return new _D(e)},2);
/**
* @license
* Copyright 2019 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*/
var MD=``+new URL(`tfjs-backend-wasm-BIA8cDLW.wasm`,self.location.href).href,ND=``+new URL(`tfjs-backend-wasm-simd-BrM3BlWr.wasm`,self.location.href).href,PD=``+new URL(`tfjs-backend-wasm-threaded-simd-BLCcuiBu.wasm`,self.location.href).href,FD=``+new URL(`g170-b6c96-s175395328-d26788732.bin-BQZ0AOs9.gz`,self.location.href).href;let ID=e=>e===1?2:1,LD=e=>e.join(``);function RD(e,t){let n=e%t,r=Math.floor(e/t);return[n>0?e-1:-1,n+1<t?e+1:-1,r>0?e-t:-1,r+1<t?e+t:-1].filter(e=>e>=0)}function zD(e,t){let n=[],r=Array(e.length).fill(-1);return e.forEach((i,a)=>{if(!i||r[a]!==-1)return;let o={id:n.length,team:i,cells:[],liberties:new Set},s=[a];for(r[a]=o.id;s.length;){let n=s.pop();o.cells.push(n);for(let a of RD(n,t))e[a]?e[a]===i&&r[a]===-1&&(r[a]=o.id,s.push(a)):o.liberties.add(a)}n.push(o)}),{units:n,unitAt:r}}function BD(e,t){let{units:n,unitAt:r}=zD(e,t),i=[],a=new Set;e.forEach((o,s)=>{if(o||a.has(s))return;let c=[],l=new Set,u=[s];for(a.add(s);u.length;){let n=u.pop();c.push(n);for(let i of RD(n,t))e[i]?l.add(r[i]):a.has(i)||(a.add(i),u.push(i))}let d=new Set([...l].map(e=>n[e].team));i.push({cells:c,boundary:l,team:d.size===1?[...d][0]:0,flag:!1,guard:!1})});let o=new Set,s=new Set,c=new Set;for(let i of[1,2]){let a=[],l=new Set;e.forEach((o,s)=>{if(o===i||l.has(s))return;let c=[],u=new Set,d=[s],f=!1;for(l.add(s);d.length;){let n=d.pop();c.push(n),e[n]&&(f=!0);for(let a of RD(n,t))e[a]===i?u.add(r[a]):l.has(a)||(l.add(a),d.push(a))}!f&&u.size&&a.push({cells:c,boundary:u,vital:new Set([...u].filter(e=>c.every(t=>n[e].liberties.has(t))))})}),a.filter(e=>e.vital.size).forEach(e=>e.cells.forEach(e=>c.add(e)));let u=new Set(n.filter(e=>e.team===i).map(e=>e.id)),d=a;for(;;){let e=[...u].filter(e=>d.filter(t=>t.vital.has(e)).length<2);if(!e.length)break;e.forEach(e=>u.delete(e)),d=d.filter(e=>[...e.boundary].every(e=>u.has(e)))}u.forEach(e=>o.add(e)),d.filter(e=>[...e.vital].some(e=>u.has(e))).forEach(e=>e.cells.forEach(e=>s.add(e)))}return i.forEach(e=>{e.guard=e.cells.every(e=>s.has(e)),e.flag=e.guard||e.cells.every(e=>c.has(e))}),{units:n,unitAt:r,regions:i,guards:s,fortresses:o}}function VD(e,t,n=BD(e.board,e.size)){if(e.ended)return`ended`;if(!Number.isInteger(t)||t<0||t>=e.board.length||e.board[t])return`occupied`;if(n.guards.has(t))return`guard`;let r=e.board.slice();r[t]=e.turn;for(let t of zD(r,e.size).units)t.team!==e.turn&&!t.liberties.size&&t.cells.forEach(e=>{r[e]=0});let i=zD(r,e.size);return i.units[i.unitAt[t]].liberties.size?e.seen.includes(LD(r))?`superko`:r:`suicide`}function HD(e){if(e.ended)return[];let t=BD(e.board,e.size);return e.board.flatMap((n,r)=>!n&&Array.isArray(VD(e,r,t))?[r]:[])}function UD(e){let t={...e,automatic:[]};for(;!t.ended&&!HD(t).length;)t={...t,turn:ID(t.turn),passes:t.passes+1,automatic:[...t.automatic,t.turn]},t.ended=t.passes>=2;return t}function WD(e,t){let n=e.position;if(n.ended)return`ended`;let r=t===null?n.board:VD(n,t);if(!Array.isArray(r))return r;let i=[...n.used];t!==null&&i[n.turn-1]++;let a=t===null?n.passes+1:0;return{position:UD({...n,board:r,used:i,turn:ID(n.turn),passes:a,ended:a>=2,last:t===null?n.last:t,seen:t===null?n.seen:[...n.seen,LD(r)]}),history:[...e.history,n]}}function GD(e,t){let n=[0,0];return e.forEach(e=>{e&&n[e-1]++}),BD(e,t).regions.forEach(e=>{e.team&&(n[e.team-1]+=e.cells.length)}),n}let Q=19,$=Q*Q,KD=$;function qD(e){return 3-e}let JD=new Int16Array($),YD=new Int8Array($),XD=new Int16Array($*4),ZD=new Int32Array($),QD=new Int32Array($),$D=0,eO=new Int16Array($),tO=new Int16Array($);function nO(e,t,n,r){$D++;let i=$D,a=0,o=0,s=0;for(ZD[t]=i,tO[o++]=t;o>0;){let t=tO[--o];eO[a++]=t;let c=JD[t],l=YD[t];for(let t=0;t<l;t++){let a=XD[c+t],l=e[a];l===0?s<r&&QD[a]!==i&&(QD[a]=i,s++):l===n&&ZD[a]!==i&&(ZD[a]=i,tO[o++]=a)}}return{groupLen:a,liberties:s}}let rO=new Int32Array($),iO=0,aO=new Int32Array($),oO=0,sO=new Int16Array($),cO=new Int16Array($),lO=new Uint8Array($),uO=new Int16Array($),dO=new Uint8Array($),fO=new Int16Array($),pO=new Int16Array($),mO=new Int16Array($),hO=($+1)/2+1|0,gO=new Int16Array(hO),_O=new Uint16Array(hO),vO=new Uint8Array(hO),yO=new Uint8Array(hO),bO=new Uint8Array(hO),xO=new Int16Array(hO*4),SO=new Int16Array($),CO=new Int16Array($),wO=new Uint8Array($),TO=new Int16Array($),EO=e=>{Q=e,$=Q*Q,KD=$,JD=new Int16Array($),YD=new Int8Array($),XD=new Int16Array($*4);let t=0;for(let e=0;e<Q;e++)for(let n=0;n<Q;n++){let r=e*Q+n;JD[r]=t;let i=0;n>0&&(XD[t++]=r-1,i++),n+1<Q&&(XD[t++]=r+1,i++),e>0&&(XD[t++]=r-Q,i++),e+1<Q&&(XD[t++]=r+Q,i++),YD[r]=i}ZD=new Int32Array($),QD=new Int32Array($),eO=new Int16Array($),tO=new Int16Array($),rO=new Int32Array($),aO=new Int32Array($),$D=0,iO=0,oO=0,sO=new Int16Array($),cO=new Int16Array($),lO=new Uint8Array($),uO=new Int16Array($),dO=new Uint8Array($),fO=new Int16Array($),pO=new Int16Array($),mO=new Int16Array($),hO=($+1)/2+1|0,gO=new Int16Array(hO),_O=new Uint16Array(hO),vO=new Uint8Array(hO),yO=new Uint8Array(hO),bO=new Uint8Array(hO),xO=new Int16Array(hO*4),SO=new Int16Array($),CO=new Int16Array($),wO=new Uint8Array($),TO=new Int16Array($),HO=$*3/2+2|0,WO={bufMoves:new Int16Array(UO),moveListStarts:new Int32Array(HO),moveListLens:new Int32Array(HO),moveListCur:new Int32Array(HO),recordMoves:new Int16Array(HO),recordPlayers:new Uint8Array(HO),recordKoPointBefore:new Int16Array(HO),recordCaptureStart:new Int32Array(HO),tmpKoPointBefore:new Int16Array(1),tmpCaptureStart:new Int32Array(1),captureStack:[]},GO=new Int32Array($),qO=new Int32Array($),YO=new Int16Array($),XO=new Int32Array($),QO=new Int32Array($),KO=0,JO=0,ZO=0,$O=0,mk={copyPos:{stones:new Uint8Array($),koPoint:-1},groupStones:new Int16Array($),workingMoves:[]}};function DO(e){let t=Math.max(2,Math.floor(e));t!==Q&&EO(t)}function OO(e,t){if(t.length!==$)throw Error(`computeLibertyMapInto: expected out length ${$}, got ${t.length}`);t.fill(0),oO++;let n=oO;for(let r=0;r<$;r++){let i=e[r];if(i===0||aO[r]===n)continue;let{groupLen:a,liberties:o}=nO(e,r,i,4),s=o>=4?4:o;for(let e=0;e<a;e++){let r=eO[e];t[r]=s,aO[r]=n}}return t}function kO(e){return OO(e,new Uint8Array($))}function AO(e){uO.fill(-1);let t=0,n=0;for(let r=0;r<$;r++){let i=e[r];if(i===0||uO[r]!==-1)continue;let a=t++;dO[a]=i,fO[a]=n;let o=0;tO[o++]=r,uO[r]=a;let s=0;for(;o>0;){let t=tO[--o];mO[n++]=t,s++;let r=JD[t],c=YD[t];for(let t=0;t<c;t++){let n=XD[r+t];e[n]===i&&uO[n]===-1&&(uO[n]=a,tO[o++]=n)}}pO[a]=s}return t}function jO(e,t,n){let r=JD[t],i=YD[t];for(let t=0;t<i;t++)if(e[XD[r+t]]===n)return!0;return!1}function MO(e,t,n,r){let i=JD[t],a=YD[t];for(let t=0;t<a;t++){let a=XD[i+t];if(e[a]===n&&uO[a]===r)return!0}return!1}function NO(e){let{stones:t,numGroups:n,plaColor:r,safeBigTerritories:i,unsafeBigTerritories:a,isMultiStoneSuicideLegal:o,result:s}=e,c=qD(r);sO.fill(-1),lO.fill(0);let l=0,u=0,d=!1,f=(e,n)=>{let i=e,a=0,s=1;SO[0]=e,sO[e]=n;let l=vO[n]>0;for(;a!==s;){let e=SO[a++];if(l&&(o||t[e]===0)){let i=_O[n],a=vO[n],o=0;for(let n=0;n<a;n++){let a=xO[i+n];MO(t,e,r,a)&&(xO[i+o]=a,o++)}vO[n]=o,l=o>0}yO[n]<2&&!jO(t,e,r)&&(yO[n]=yO[n]+1),t[e]===c&&(bO[n]=1),cO[e]=i,i=e;let u=JD[e],d=YD[e];for(let e=0;e<d;e++){let r=XD[u+e],i=t[r];(i===0||i===c)&&sO[r]===-1&&(sO[r]=n,SO[s++]=r)}}return i};for(let e=0;e<$;e++){if(sO[e]!==-1)continue;let n=t[e];if(n!==0){n===r&&(d=!0);continue}let i=l++;gO[i]=e,_O[i]=u,vO[i]=0,yO[i]=0,bO[i]=0;let a=0;{let n=JD[e],i=YD[e];for(let e=0;e<i;e++){let i=XD[n+e];if(t[i]!==r)continue;let o=uO[i],s=!1;for(let e=0;e<a;e++)if(xO[u+e]===o){s=!0;break}if(!s&&(xO[u+a]=o,a++,a>=4))break}}vO[i]=a;let o=f(e,i);cO[e]=o,u+=vO[i]}let p=0;for(let e=0;e<n;e++)dO[e]===r&&(CO[p++]=e,wO[e]=0,TO[e]=0);for(let e=0;e<l;e++){let t=_O[e],n=vO[e];for(let e=0;e<n;e++){let n=xO[t+e];TO[n]=TO[n]+1}}for(;;){let e=!1;for(let n=0;n<p;n++){let r=CO[n];if(wO[r]||TO[r]>=2)continue;wO[r]=1,e=!0;let i=fO[r],a=pO[r];for(let e=0;e<a;e++){let n=mO[i+e],r=JD[n],a=YD[n];for(let e=0;e<a;e++){let n=XD[r+e],i=sO[n];if(i<0)continue;let a=gO[i];if(lO[a])continue;let o=t[n];if(o!==0&&o!==c)continue;lO[a]=1;let s=_O[i],l=vO[i];for(let e=0;e<l;e++){let t=xO[s+e];TO[t]=TO[t]-1}}}}if(!e)break}for(let e=0;e<p;e++){let t=CO[e];if(wO[t])continue;let n=fO[t],i=pO[t];for(let e=0;e<i;e++)s[mO[n+e]]=r}for(let e=0;e<l;e++){let t=gO[e],n=yO[e]<=1&&!lO[t]&&d;if(n||=i&&!bO[e]&&!lO[t]&&d,n){let e=t;do s[e]=r,e=cO[e];while(e!==t)}else if(a&&!bO[e]&&d){let e=t;do s[e]===0&&(s[e]=r),e=cO[e];while(e!==t)}}}function PO(e,t=!1){return FO(e,new Uint8Array($),t)}function FO(e,t,n=!1){if(t.length!==$)throw Error(`computeAreaMapV7KataGoInto: expected out length ${$}, got ${t.length}`);t.fill(0);let r=AO(e);NO({stones:e,numGroups:r,plaColor:1,safeBigTerritories:!0,unsafeBigTerritories:!0,isMultiStoneSuicideLegal:n,result:t}),NO({stones:e,numGroups:r,plaColor:2,safeBigTerritories:!0,unsafeBigTerritories:!0,isMultiStoneSuicideLegal:n,result:t});for(let n=0;n<$;n++)t[n]===0&&(t[n]=e[n]);return t}function IO(e,t){let n=e[t];return n===0?0:nO(e,t,n,2).liberties}let LO=new Uint8Array($),RO=new Int16Array($),zO=new Uint8Array($);function BO(e,t,n={}){if(t.length!==$)throw Error(`computeIndependentLifeAreaInto: expected out length ${$}, got ${t.length}`);zO.length!==$&&(zO=new Uint8Array($));let r=zO;FO(e,r,n.isMultiStoneSuicideLegal??!1),t.fill(0);let i=LO.subarray(0,$);i.fill(0);let a=RO;for(let t=0;t<$;t++){let n=r[t];if(n===0||i[t])continue;let o=!1;if(e[t]===n&&IO(e,t)===1&&(o=!0),!o){let n=JD[t],i=YD[t];for(let t=0;t<i;t++){let i=XD[n+t];if(e[i]===0&&r[i]===0){o=!0;break}}}if(!o)continue;let s=0,c=0;for(i[t]=1,a[c++]=t;s!==c;){let e=a[s++],t=JD[e],o=YD[e];for(let e=0;e<o;e++){let o=XD[t+e];r[o]===n&&!i[o]&&(i[o]=1,a[c++]=o)}}}let o=0;for(let e=0;e<$;e++){let n=r[e];if(n===0||i[e]||t[e]===n)continue;o+=n===2?1:-1,t[e]=n;let s=0,c=0;for(a[c++]=e;s!==c;){let e=a[s++],i=JD[e],o=YD[e];for(let e=0;e<o;e++){let o=XD[i+e];r[o]===n&&t[o]!==n&&(t[o]=n,a[c++]=o)}}}if(n.keepTerritories)for(let i=0;i<$;i++){let a=r[i];if(a===0||a===e[i])continue;let o=!1;if(n.excludeTerritoryAdjacentToAtari&&e[i]===0){let t=JD[i],n=YD[i];for(let r=0;r<n;r++){let n=XD[t+r];if(e[n]===a&&IO(e,n)===1){o=!0;break}}}o||(t[i]=a)}if(n.keepStones)for(let n=0;n<$;n++){let i=r[n];i!==0&&i===e[n]&&(t[n]=i)}return{whiteMinusBlackIndependentLifeRegionCount:o}}function VO(e,t={}){let n=new Uint8Array($),{whiteMinusBlackIndependentLifeRegionCount:r}=BO(e,n,t);return{area:n,whiteMinusBlackIndependentLifeRegionCount:r}}let HO=$*3/2+2|0,UO=8192,WO={bufMoves:new Int16Array(UO),moveListStarts:new Int32Array(HO),moveListLens:new Int32Array(HO),moveListCur:new Int32Array(HO),recordMoves:new Int16Array(HO),recordPlayers:new Uint8Array(HO),recordKoPointBefore:new Int16Array(HO),recordCaptureStart:new Int32Array(HO),tmpKoPointBefore:new Int16Array(1),tmpCaptureStart:new Int32Array(1),captureStack:[]},GO=new Int32Array($),KO=0,qO=new Int32Array($),JO=0,YO=new Int16Array($),XO=new Int32Array($),ZO=0,QO=new Int32Array($),$O=0;function ek(e,t){if(e===t)return!1;let n=e-t;return n===1||n===-1?(e/Q|0)==(t/Q|0):n===Q||n===-Q}function tk(e,t){let n=0,r=JD[t],i=YD[t];for(let t=0;t<i;t++)e[XD[r+t]]===0&&n++;return n}function nk(e,t,n,r,i,a){$D++;let o=$D,s=0,c=0,l=0;for(ZD[t]=o,tO[s++]=t;s>0;){let t=tO[--s];eO[c++]=t;let u=JD[t],d=YD[t];for(let t=0;t<d;t++){let c=XD[u+t],d=e[c];d===0?l<a&&QD[c]!==o&&(QD[c]=o,r[i+l]=c,l++):d===n&&ZD[c]!==o&&(ZD[c]=o,tO[s++]=c)}}return l}function rk(e,t,n){let r=e[t];return r===0?0:nO(e,t,r,n).liberties}function ik(e,t,n){if(e[t]!==0)return!1;let r=qD(n),i=-1,a=JD[t],o=YD[t];for(let t=0;t<o;t++){let n=XD[a+t];if(e[n]!==r)return!1;if(rk(e,n,2)===1){if(i!==-1)return!1;i=n}}if(i===-1)return!1;let s=nO(e,i,r,2);return s.liberties===1&&s.groupLen===1}function ak(e,t){let n=e[t];if(n===0)return!1;let r=qD(n),i=nO(e,t,n,2);for(let e=0;e<i.groupLen;e++)YO[e]=eO[e];for(let t=0;t<i.groupLen;t++){let n=YO[t],i=JD[n],a=YD[n];for(let t=0;t<a;t++){let n=XD[i+t];if(e[n]===r&&rk(e,n,2)===1)return!0}}return!1}function ok(e,t,n){let r=0,i=JD[t],a=YD[t];for(let t=0;t<a;t++){let a=XD[i+t];if(e[a]!==n)continue;let o=rk(e,a,20);o>1&&(r+=o*2-3)}return r}function sk(e,t,n){let r=qD(n),i=0,a=0,o=0,s=0,c=0,l=JD[t],u=YD[t];for(let t=0;t<u;t++){let u=XD[l+t],d=e[u];if(d===0)i++;else if(d===r){let t=nO(e,u,r,2);t.liberties===1&&(a++,o+=t.groupLen)}else if(d===n){let t=rk(e,u,20)-1;s+=t,t>c&&(c=t)}}return{lowerBound:a+(c>i?c:i),upperBound:i+o+s}}function ck(e,t,n,r){if(e[t]!==0)return 0;let i=qD(n);$O++;let a=$O,o=new Int16Array(16),s=0,c=e=>{for(let t=0;t<s;t++)if(o[t]===e)return!1;return o[s++]=e,!0},l=JD[t],u=YD[t];for(let t=0;t<u;t++){let n=XD[l+t],o=e[n];if(o===0){if(c(n),s>=r)return r}else if(o===i){let t=nO(e,n,i,2);if(t.liberties===1){if(c(n),s>=r)return r;for(let e=0;e<t.groupLen;e++)QO[eO[e]]=a}}}ZO++;let d=ZO,f=t=>{let n=e[t];return n===0||n===i&&QO[t]===a};for(let i=0;i<u;i++){let a=XD[l+i];if(e[a]!==n||XO[a]===d)continue;let o=nO(e,a,n,r);for(let e=0;e<o.groupLen;e++)XO[eO[e]]=d;for(let e=0;e<o.groupLen;e++){let n=eO[e],i=JD[n],a=YD[n];for(let e=0;e<a;e++){let n=XD[i+e];if(n!==t&&f(n)&&(c(n),s>=r))return r}}}return s}function lk(e,t,n,r){let i=e[t];if(i===0)return 0;let a=qD(i);JO++;let o=JO,s=nO(e,t,i,2);for(let e=0;e<s.groupLen;e++)YO[e]=eO[e];let c=0;for(let t=0;t<s.groupLen;t++){let i=YO[t],s=JD[i],l=YD[i];for(let t=0;t<l;t++){let i=XD[s+t];if(e[i]!==a||qO[i]===o)continue;let l=nk(e,i,a,n,r+c,2),u=nO(e,i,a,2).groupLen;for(let e=0;e<u;e++)qO[eO[e]]=o;if(l===1&&(c+=1,r+c>=n.length))return c}}return c}function uk(e,t,n,r,i,a){let o=a.length,s=qD(n);t!==KD&&(e.stones[t]=0);for(let t=i;t<o;t++){let n=a[t];e.stones[n]=s}a.length=i,e.koPoint=r}function dk(e,t,n,r,i,a,o){let s=e.koPoint,c=r.length;if(i[o]=s,a[o]=c,t===KD)return e.koPoint=-1,!0;if(t<0||t>=$||e.stones[t]!==0||e.koPoint===t)return!1;let l=qD(n);e.stones[t]=n;let u=0,d=-1;iO++;let f=iO,p=JD[t],m=YD[t];for(let t=0;t<m;t++){let n=XD[p+t];if(e.stones[n]!==l||rO[n]===f)continue;let{groupLen:i,liberties:a}=nO(e.stones,n,l,1);for(let e=0;e<i;e++)rO[eO[e]]=f;if(a===0){for(let t=0;t<i;t++){let n=eO[t];e.stones[n]=0,r.push(n)}u+=i,u===1&&i===1&&(d=eO[0]),u>1&&(d=-1)}}let h=nO(e.stones,t,n,2);return h.liberties===0?(uk(e,t,n,s,c,r),!1):(e.koPoint=u===1&&d>=0&&h.groupLen===1&&h.liberties===1?d:-1,!0)}function fk(e,t,n,r){if(t<0||t>=$)return!1;let i=e.stones[t];if(i!==1&&i!==2)return!1;let a=rk(e.stones,t,3);if(a>2||n&&a>1)return!1;let o=i,s=qD(o),c=e.koPoint;n&&(e.koPoint=-1);let{bufMoves:l,moveListStarts:u,moveListLens:d,moveListCur:f,recordMoves:p,recordPlayers:m,recordKoPointBefore:h,recordCaptureStart:g,captureStack:_}=r,v=HO,y=0,b=0;f[0]=-1,u[0]=0,d[0]=0;let x=!1,S=!1;for(;;){if(y<=-1)return e.koPoint=c,x;if(y>=v-1){x=!0,S=!0,y--;continue}if(b>=25e3){for(let t=y-1;t>=0;t--)uk(e,p[t],m[t],h[t],g[t],_);return e.koPoint=c,!1}let r=n&&y%2==0||!n&&y%2==1;if(f[y]===-1){let n=rk(e.stones,t,3);if(!r&&n<=1){x=!0,S=!0,y--;continue}if(!r&&n>=3){x=!1,S=!0,y--;continue}if(r&&n>=2){x=!1,S=!0,y--;continue}if(r&&e.koPoint!==-1){x=!1,S=!0,y--;continue}let i=u[y],a=0;if(r){if(a=lk(e.stones,t,l,i),a+=nk(e.stones,t,o,l,i+a,1),a<=0){x=!0,S=!0,y--;continue}let n=l[i+a-1],r=sk(e.stones,n,o);if(r.lowerBound>=3){x=!1,S=!0,y--;continue}if(a===1&&r.upperBound<=1){x=!0,S=!0,y--;continue}}else{if(a+=nk(e.stones,t,o,l,i,2),a!==2){x=!1,S=!0,y--;continue}let n=tk(e.stones,l[i]),r=tk(e.stones,l[i+1]);if(n===0&&r===0&&ik(e.stones,l[i],s)&&ik(e.stones,l[i+1],s)&&ck(e.stones,l[i],o,3)<=2&&ck(e.stones,l[i+1],o,3)<=2&&!ak(e.stones,t)){x=!0,S=!0,y--;continue}if(!ek(l[i],l[i+1])){if(n>=3&&r>=3){x=!1,S=!0,y--;continue}n>=3?a=1:r>=3&&(l[i]=l[i+1],a=1)}if(a>1&&(n=n*2+ok(e.stones,l[i],o),r=r*2+ok(e.stones,l[i+1],o),r>n)){let e=l[i];l[i]=l[i+1],l[i+1]=e}}d[y]=a,f[y]=0}else{if(S&&uk(e,p[y],m[y],h[y],g[y],_),r&&!x){S=!0,y--;continue}if(!r&&x){S=!0,y--;continue}f[y]=f[y]+1|0}if(f[y]>=d[y]){x=r,S=!0,y--;continue}let i=l[u[y]+f[y]],a=r?o:s;if(p[y]=i,m[y]=a,!dk(e,i,a,_,h,g,y)){x=r,S=!1;continue}b++,y++,f[y]=-1,u[y]=u[y-1]+d[y-1],d[y]=0}}function pk(e,t,n,r){if(t<0||t>=$)return!1;let i=e.stones[t];if(i!==1&&i!==2||rk(e.stones,t,3)!==2)return!1;let a=i,o=qD(a),s=new Int16Array(2);if(nk(e.stones,t,a,s,0,2)!==2)return!1;let c=s[0],l=s[1],u=!1,d=!1;return r.length=0,n.captureStack.length=0,dk(e,c,o,n.captureStack,n.tmpKoPointBefore,n.tmpCaptureStart,0)&&(u=fk(e,t,!0,n),uk(e,c,o,n.tmpKoPointBefore[0],n.tmpCaptureStart[0],n.captureStack)),dk(e,l,o,n.captureStack,n.tmpKoPointBefore,n.tmpCaptureStart,0)&&(d=fk(e,t,!0,n),uk(e,l,o,n.tmpKoPointBefore[0],n.tmpCaptureStart[0],n.captureStack)),u||d?(u&&r.push(c),d&&r.push(l),!0):!1}let mk={copyPos:{stones:new Uint8Array($),koPoint:-1},groupStones:new Int16Array($),workingMoves:[]};function hk(e){let{stones:t,koPoint:n,currentPlayer:r,outLadderedStones:i,outLadderWorkingMoves:a}=e;i.fill(0),a.fill(0);let o=qD(r);KO++;let s=KO,c=mk.copyPos,l=mk.groupStones,u=mk.workingMoves;for(let e=0;e<$;e++){let r=t[e];if(r===0||GO[e]===s)continue;let d=nO(t,e,r,3);for(let e=0;e<d.groupLen;e++)GO[eO[e]]=s;if(d.liberties!==1&&d.liberties!==2)continue;l.set(eO.subarray(0,d.groupLen)),c.stones.set(t),c.koPoint=n,WO.captureStack.length=0;let f=!1;if(u.length=0,f=d.liberties===1?fk(c,e,!0,WO):pk(c,e,WO,u),f){for(let e=0;e<d.groupLen;e++)i[l[e]]=1;if(d.liberties===2&&r===o&&u.length>0)for(let e=0;e<u.length;e++)a[u[e]]=1}}}function gk(e){let t=new Uint8Array($),n=new Uint8Array($);return hk({...e,outLadderedStones:t,outLadderWorkingMoves:n}),{ladderedStones:t,ladderWorkingMoves:n}}function _k(e){let{stones:t,koPoint:n,outLadderedStones:r}=e;r.fill(0),KO++;let i=KO,a=mk.copyPos,o=mk.groupStones,s=mk.workingMoves;for(let e=0;e<$;e++){let c=t[e];if(c===0||GO[e]===i)continue;let l=nO(t,e,c,3);for(let e=0;e<l.groupLen;e++)GO[eO[e]]=i;if(l.liberties!==1&&l.liberties!==2)continue;o.set(eO.subarray(0,l.groupLen)),a.stones.set(t),a.koPoint=n,WO.captureStack.length=0,s.length=0;let u=!1;if(u=l.liberties===1?fk(a,e,!0,WO):pk(a,e,WO,s),u)for(let e=0;e<l.groupLen;e++)r[o[e]]=1}}function vk(e){let t=new Uint8Array($);return _k({...e,outLadderedStones:t}),t}EO(Q);let yk={japanese:{id:`japanese`,label:`Japanese`,summary:`Territory scoring, seki tax, simple ko.`,scoring:`territory`,ko:`simple`,tax:`seki`,multiStoneSuicideLegal:!1,hasButton:!1,friendlyPassOk:!1,handicapBonus:`zero`,defaultKomi:6.5,sgf:`Japanese`},korean:{id:`korean`,label:`Korean`,summary:`Same as Japanese: territory scoring, seki tax, simple ko.`,scoring:`territory`,ko:`simple`,tax:`seki`,multiStoneSuicideLegal:!1,hasButton:!1,friendlyPassOk:!1,handicapBonus:`zero`,defaultKomi:6.5,sgf:`Korean`},chinese:{id:`chinese`,label:`Chinese`,summary:`Area scoring, simple ko, White gets a point per handicap stone.`,scoring:`area`,ko:`simple`,tax:`none`,multiStoneSuicideLegal:!1,hasButton:!1,friendlyPassOk:!0,handicapBonus:`n`,defaultKomi:7.5,sgf:`Chinese`},aga:{id:`aga`,label:`AGA`,summary:`Area scoring, situational superko, White gets N−1 for handicap.`,scoring:`area`,ko:`situational`,tax:`none`,multiStoneSuicideLegal:!1,hasButton:!1,friendlyPassOk:!0,handicapBonus:`n-minus-one`,defaultKomi:7.5,sgf:`AGA`},"new-zealand":{id:`new-zealand`,label:`New Zealand`,summary:`Area scoring, situational superko, suicide is legal.`,scoring:`area`,ko:`situational`,tax:`none`,multiStoneSuicideLegal:!0,hasButton:!1,friendlyPassOk:!0,handicapBonus:`zero`,defaultKomi:7,sgf:`New Zealand`},"stone-scoring":{id:`stone-scoring`,label:`Ancient Chinese`,summary:`Area scoring with a two-point tax per living group (stone scoring).`,scoring:`area`,ko:`simple`,tax:`all`,multiStoneSuicideLegal:!1,hasButton:!1,friendlyPassOk:!0,handicapBonus:`zero`,defaultKomi:7.5,sgf:`Stone Scoring`},"tromp-taylor":{id:`tromp-taylor`,label:`Tromp-Taylor`,summary:`Area scoring, positional superko, suicide is legal.`,scoring:`area`,ko:`positional`,tax:`none`,multiStoneSuicideLegal:!0,hasButton:!1,friendlyPassOk:!1,handicapBonus:`zero`,defaultKomi:7.5,sgf:`Tromp-Taylor`}};yk.japanese,yk.chinese,yk.korean,yk.aga,yk[`new-zealand`],yk[`tromp-taylor`],yk[`stone-scoring`];let bk=e=>yk[e]??yk.japanese,xk=e=>bk(e).multiStoneSuicideLegal,Sk=e=>{let t=bk(e);return t.scoring===`area`?t.tax===`none`?`pass-alive`:`independent-life`:`none`};function Ck(e){let{recentMoves:t,currentPlayer:n}=e,r=bk(e.rules),i=t.length-1,a=t[i]?.move===KD,o=a&&t[i-1]?.move===KD,s=r.scoring===`area`&&a,c=!1;if(s&&e.enablePassingHacks&&e.areaMap){let t=n===`black`?1:2,r=3-t,i=e.selfKomi??0;for(let n of e.areaMap)i+=n===t?1:n===r?-1:0;c=i<=0}if(s&&(e.conservativePassAndIsRoot===!0||r.friendlyPassOk&&!o||c))return{turnsIncluded:0,passWouldEndPhase:!1};let l=Math.max(0,Math.min(5,e.maxHistory??5));for(let e=0;e<l&&i-e>0;e++)if(t[i-e].move===KD&&t[i-e-1].move===KD){l=Math.min(l,e+1);break}let u=0,d=n===`black`?`white`:`black`;for(let e=0;e<l;e++){let n=t[i-e];if(!n||n.player!==d)break;u++,d=d===`black`?`white`:`black`}return{turnsIncluded:u,passWouldEndPhase:a}}let wk=e=>e===`black`?`white`:`black`,Tk=(e,t,n)=>(t*Q+e)*22+n;function Ek(e){return e===`black`?1:2}function Dk(e){let{stones:t,koPoint:n,currentPlayer:r,recentMoves:i,komi:a}=e,o=e.rules??`japanese`,s=e.playoutDoublingAdvantage??0,c=r,l=wk(c),u=Ek(c),d=Ek(l),f=e.outSpatial,p=e.outGlobal;f.fill(0),p.fill(0);for(let e=0;e<Q*Q;e++)f[e*22+0]=1;if(n>=0&&n<Q*Q){let e=n%Q,t=n/Q|0;f[Tk(e,t,6)]=1}if(e.superkoBanned)for(let t=0;t<Q*Q;t++)e.superkoBanned[t]&&(f[t*22+6]=1);let m=e.libertyMap??kO(t);for(let e=0;e<Q;e++)for(let n=0;n<Q;n++){let r=e*Q+n,i=t[r];if(i===0)continue;i===u?f[Tk(n,e,1)]=1:i===d&&(f[Tk(n,e,2)]=1);let a=m[r];a===1?f[Tk(n,e,3)]=1:a===2?f[Tk(n,e,4)]=1:a===3&&(f[Tk(n,e,5)]=1)}if(e.ladderedStones||e.prevLadderedStones||e.prevPrevLadderedStones||e.ladderWorkingMoves){let t=e.ladderedStones,n=e.prevLadderedStones,r=e.prevPrevLadderedStones,i=e.ladderWorkingMoves;for(let e=0;e<Q;e++)for(let a=0;a<Q;a++){let o=e*Q+a;t&&t[o]&&(f[Tk(a,e,14)]=1),n&&n[o]&&(f[Tk(a,e,15)]=1),r&&r[o]&&(f[Tk(a,e,16)]=1),i&&i[o]&&(f[Tk(a,e,17)]=1)}}let h=e.drawEquivalentWinsForWhite??.5,g=a+(Math.trunc(a)===a?h-.5:0),_=c===`white`?g:-g,v=Sk(o),y=v!==`none`,b;if(y){b=e.areaMap??(v===`independent-life`?VO(t,{keepStones:!0,isMultiStoneSuicideLegal:xk(o)}).area:PO(t,xk(o)));for(let e=0;e<Q;e++)for(let t=0;t<Q;t++){let n=e*Q+t,r=b[n];r===u?f[Tk(t,e,18)]=1:r===d&&(f[Tk(t,e,19)]=1)}}let x=e.historyFeatures??Ck({recentMoves:i,currentPlayer:r,rules:o,selfKomi:_,areaMap:b,conservativePassAndIsRoot:e.conservativePassAndIsRoot,enablePassingHacks:e.enablePassingHacks,maxHistory:e.maxHistory}),S=[9,10,11,12,13],C=[0,1,2,3,4],w=[l,c,l,c,l];for(let e=0;e<x.turnsIncluded;e++){let t=i[i.length-1-e];if(!t||t.player!==w[e])break;if(t.move===KD)p[C[e]]=1;else{let n=t.move%Q,r=t.move/Q|0;f[Tk(n,r,S[e])]=1}}let T=Q*Q+20,E=Math.max(-T,Math.min(T,_));p[5]=E/20;let D=bk(o);if(D.ko===`positional`?(p[6]=1,p[7]=.5):D.ko===`situational`&&(p[6]=1,p[7]=-.5),D.multiStoneSuicideLegal&&(p[8]=1),D.scoring===`territory`&&(p[9]=1),D.tax===`seki`?p[10]=1:D.tax===`all`&&(p[10]=1,p[11]=1),D.hasButton&&(p[17]=1),p[14]=+!!x.passWouldEndPhase,s!==0&&(p[15]=1,p[16]=.5*s),D.scoring===`area`){let e=Q*Q%2==0,t;t=e?Math.floor(E/2)*2:Math.floor((E-1)/2)*2+1;let n=E-t;n<0&&(n=0),n>2&&(n=2);let r;r=n<.5?n:n<1.5?1-n:n-2,p[18]=r}}let Ok=(e,t)=>(t-1-Math.floor(e/t))*t+e%t,kk=(e,t)=>Uint8Array.from(e,(n,r)=>e[Ok(r,t)]);function Ak(e){let t=[...e.history,e.position],n=[],r=[t[0].board];for(let e=1;e<t.length;e++){let i=t[e-1],a=t[e],o=a.used[0]+a.used[1]>i.used[0]+i.used[1];n.push({move:o?Ok(a.last,a.size):a.size**2,player:i.turn===1?`black`:`white`}),r.push(a.board);for(let e of a.automatic)n.push({move:a.size**2,player:e===1?`black`:`white`}),r.push(a.board)}return{moves:n,boards:r}}function jk(e,t,n){if(t<=0)return-1;let r=e[t-1],i=e[t],a=[],o=[];for(let e=0;e<i.length;e++)r[e]&&!i[e]&&a.push(e),!r[e]&&i[e]&&o.push(e);if(a.length!==1||o.length!==1)return-1;let{units:s,unitAt:c}=zD(i,n),l=s[c[o[0]]];return l.cells.length===1&&l.liberties.size===1?Ok(a[0],n):-1}function Mk(e){let t=e.position,n=t.size;if(![7,9,11].includes(n))throw Error(`Unsupported Jin-Go board size.`);if(t.ended)throw Error(`Terminal positions must be scored by Jin-Go, not the network.`);DO(n);let r=kk(t.board,n),{moves:i,boards:a}=Ak(e),o={...BD(t.board,n),guards:new Set},s=new Uint8Array(n**2);for(let e=0;e<t.board.length;e++)VD(t,e,o)===`superko`&&(s[Ok(e,n)]=1);let c=jk(a,a.length-1,n),l=gk({stones:r,koPoint:c,currentPlayer:t.turn}),u=kk(a[Math.max(0,a.length-2)],n),d=kk(a[Math.max(0,a.length-3)],n),f=new Float32Array(n*n*22),p=new Float32Array(19);return Dk({stones:r,koPoint:c,superkoBanned:s,currentPlayer:t.turn===1?`black`:`white`,recentMoves:i,komi:0,rules:`tromp-taylor`,conservativePassAndIsRoot:!1,enablePassingHacks:!1,areaMap:PO(r,!1),ladderedStones:l.ladderedStones,ladderWorkingMoves:l.ladderWorkingMoves,prevLadderedStones:vk({stones:u,koPoint:jk(a,a.length-2,n)}),prevPrevLadderedStones:vk({stones:d,koPoint:jk(a,a.length-3,n)}),outSpatial:f,outGlobal:p}),p[8]=0,{size:n,spatial:f,global:p,nextPlayer:t.turn===1?`black`:`white`}}function Nk(e,t){let n=e.position,r=BD(n.board,n.size),i=n.size**2,a=[];for(let e=0;e<i;e++)Array.isArray(VD(n,e,r))&&a.push({move:e,logit:t[Ok(e,n.size)]});if(a.push({move:null,logit:t[i]}),a.some(e=>!Number.isFinite(e.logit)))throw Error(`Non-finite model policy`);let o=Math.max(...a.map(e=>e.logit)),s=a.reduce((e,t)=>e+Math.exp(t.logit-o),0);return a.map(e=>({move:e.move,probability:Math.exp(e.logit-o)/s})).sort((e,t)=>t.probability-e.probability)}let Pk=(e,t)=>{let n=GD(e.position.board,e.position.size);return n[t-1]-n[2-t]};async function Fk(e,t,{deadline:n,signal:r,maxNodes:i=2048}){let a=e.position.turn,o=0,s=()=>r?.aborted||performance.now()>=n||o>=i,c=e=>{if(e.position.ended||e.position.turn===a)return Pk(e,a);let t=1/0;for(let n of[null,...HD(e.position)]){if(s())return null;let r=WD(e,n);if(typeof r==`string`)throw Error(`Illegal pass audit reply: ${r}`);o++,t=Math.min(t,Pk(r,a))}return t};if(s())return{move:null,status:`incomplete`,nodes:o};let l=WD(e,null),u=c(l);if(u===null)return{move:null,status:`incomplete`,nodes:o};if(l.position.ended&&u>0)return{move:null,status:`complete`,passLead:u,bestLead:u,nodes:o,examined:0};let d=u,f=null,p=0;for(let n of t){if(n===null)continue;if(s())break;let t=WD(e,n);if(typeof t==`string`)throw Error(`Illegal pass audit move: ${t}`);o++;let r=c(t);if(r===null)break;p++,r>d&&(d=r,f=n),p%4==0&&await new Promise(e=>setTimeout(e,0))}return{move:f,status:p===t.filter(e=>e!==null).length?`complete`:`incomplete`,passLead:u,bestLead:d,nodes:o,examined:p}}function Ik(e){let t=GD(e.board,e.size),n=t[0]-t[1];return Math.sign(n)+.15*Math.tanh(n/e.size)}function Lk(e){return async t=>{let n=await yy(e,Mk(t)),r=n.evaluation;return{value:2*r.blackWinProb+r.blackNoResultProb-1+.15*Math.tanh(r.blackScoreLead/t.position.size),policy:Nk(t,n.logits)}}}let Rk=e=>({game:e,visits:0,sum:0,value:0,children:null}),zk=e=>e.exact??(e.visits?e.sum/e.visits:e.value);async function Bk(e,t,{maxVisits:n=128,timeMs:r=1e3,signal:i,onVisit:a}={}){if(![7,9,11].includes(e.position.size))throw Error(`Unsupported Jin-Go board size`);if(!Number.isInteger(n)||n<0||!Number.isFinite(r)||r<0)throw Error(`Invalid search budget`);let o=performance.now(),s=Rk(e),c=0,l=0,u=0,d=`visits`,f=0,p=()=>i?.aborted?`cancelled`:performance.now()-o>=r-f?`time`:null,m=(e,t)=>{if(!t.node){let n=WD(e.game,t.move);if(typeof n==`string`)throw Error(`Search generated illegal move ${t.move}: ${n}`);t.node=Rk(n),n.position.ended&&(t.node.exact=Ik(n.position))}return t.node},h=async e=>{if(e.game.position.ended)return e.exact=Ik(e.game.position),e.exact;let n=HD(e.game.position),i=n.length?await t(e.game):{value:0,policy:[{move:null,probability:1}]};if(n.length&&c++,!Number.isFinite(i.value)||Math.abs(i.value)>1.151)throw Error(`Invalid neural value`);e.value=i.value;let a=[...n,null],o=new Map;for(let{move:e,probability:t}of i.policy){if(!a.includes(e)||!Number.isFinite(t)||t<0)throw Error(`Invalid neural policy`);o.set(e,t)}let l=a.reduce((e,t)=>e+Math.max(1e-6,o.get(t)??0),0);e.children=a.map(e=>({move:e,prior:Math.max(1e-6,o.get(e)??0)/l,node:null})),e===s&&e.children.at(-1).prior>=.05&&(f=Math.min(250,r/4));let u=e.children.at(-1);return(e.game.position.passes>0||!n.length)&&m(e,u),e.value};if(e.position.ended)return{move:null,visits:0,evaluations:0,maxDepth:0,elapsedMs:performance.now()-o,stopReason:`terminal`,candidates:[]};for(;u<n;u++){let e=p();if(e){d=e;break}let t=[s],n=s;for(;n.children&&!n.game.position.ended;){let e=n.game.position.turn===1?1:-1,r,i=-1/0;for(let t of n.children){let a=t.node?.visits??0,o=(t.node&&(a||t.node.exact!==void 0)?e*zk(t.node):e*n.value-.2)+1.5*t.prior*Math.sqrt(Math.max(1,n.visits))/(1+a);o>i&&(i=o,r=t)}n=m(n,r),t.push(n)}l=Math.max(l,t.length-1);let r=n.exact??await h(n);for(let e of t)e.visits++,e.sum+=r;a?.({visits:u+1,depth:t.length-1,position:n.game.position}),(u+1)%8==0&&await new Promise(e=>setTimeout(e,0))}i?.aborted&&(d=`cancelled`);let g=e.position.turn===1?1:-1,_=(s.children??[]).map(e=>({move:e.move,prior:e.prior,visits:e.node?.visits??0,value:e.node?g*zk(e.node):null,exact:e.node?.exact!==void 0}));_.sort((e,t)=>t.visits-e.visits||t.prior-e.prior);let v=_.length?_[0].move:HD(e.position)[0]??null,y=_.find(e=>e.move===null&&e.exact),b;if(y?.value>0?v=null:v===null&&_.length&&!i?.aborted&&(b=await Fk(e,_.map(e=>e.move),{deadline:o+r,signal:i}),b.move!==null&&(v=b.move)),v===null&&y?.value<0){let e=_.find(e=>e.move!==null&&e.value!==null&&e.value>y.value);e&&(v=e.move)}return i?.aborted&&(d=`cancelled`),{move:v,visits:u,evaluations:c,maxDepth:l,elapsedMs:performance.now()-o,stopReason:d,candidates:_,passAudit:b}}let Vk={beginner:{label:`入門`,maxVisits:1,timeMs:400,candidates:12,temperature:2.5},easy:{label:`やさしい`,maxVisits:16,timeMs:450,candidates:6,temperature:1.5},normal:{label:`ふつう`,maxVisits:48,timeMs:700,candidates:3,temperature:.7},strong:{label:`強い`,maxVisits:128,timeMs:1e3,candidates:1,temperature:0}};function Hk(e){return typeof e==`string`&&Object.hasOwn(Vk,e)?e:`easy`}function Uk(e,t,n,r=Math.random){let i=Vk[Hk(n)];if(i.candidates===1||t.move===null||t.passAudit?.move!=null||t.candidates.some(e=>e.move===t.move&&e.exact&&e.value>0))return t.move;let a=e.position,o=a.turn,s=new Set(BD(a.board,a.size).regions.filter(e=>e.team===o).flatMap(e=>e.cells)),c=a.board.filter(e=>e&&e!==o).length,l=t.candidates.filter(e=>e.move!==null&&!(e.exact&&e.value<0)),u=l.filter(t=>{let n=WD(e,t.move);if(typeof n==`string`)return!1;let r=n.position.board.filter(e=>e&&e!==o).length<c,i=zD(n.position.board,a.size),l=i.units[i.unitAt[t.move]];return r||l.liberties.size>1&&!s.has(t.move)}).slice(0,i.candidates);if(!u.length)return t.move;let d=l.reduce((e,t)=>e+t.visits,0),f=u.map(e=>Math.max(1e-6,d?.5*e.prior+.5*e.visits/d:e.prior)**(1/i.temperature)),p=r()*f.reduce((e,t)=>e+t,0);for(let e=0;e<u.length;e++)if(p-=f[e],p<0)return u[e].move;return u.at(-1).move}async function Wk(e,t,n,{random:r=Math.random,signal:i}={}){let a=Vk[Hk(n)],o=await Bk(e,t,{maxVisits:a.maxVisits,timeMs:a.timeMs,signal:i});if(i?.aborted)return o;let s=performance.now(),c=Uk(e,o,n,r);return{...o,move:c,elapsedMs:o.elapsedMs+performance.now()-s}}let Gk;async function Kk(e,t){let n=e=>new URL(e,t).href;jD(1),OD({"tfjs-backend-wasm.wasm":n(MD),"tfjs-backend-wasm-simd.wasm":n(ND),"tfjs-backend-wasm-threaded-simd.wasm":n(PD)});let r=await fetch(n(FD));if(!r.ok)throw Error(`Model download failed (${r.status})`);let i=new Uint8Array(await r.arrayBuffer());for(let t of[`wasm`,`cpu`])try{if(!await ki(t))throw Error(`Backend unavailable: ${t}`);await Ai(),Gk=await vy(i),await yy(Gk,Mk(e));return}catch(e){if(Gk?.dispose(),Gk=null,t===`cpu`)throw e}}self.onmessage=async({data:e})=>{let{id:t,game:n,baseUrl:r,difficulty:i}=e;try{Gk||(self.postMessage({id:t,type:`loading`}),await Kk(n,r)),self.postMessage({id:t,type:`thinking`,backend:ji()});let e=await Wk(n,Lk(Gk),i);self.postMessage({id:t,type:`move`,move:e.move})}catch(e){self.postMessage({id:t,type:`error`,error:e.message})}}})();