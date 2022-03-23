"use strict";(self.webpackChunkneuro_symbolic_ai_toolkit_site=self.webpackChunkneuro_symbolic_ai_toolkit_site||[]).push([[7056],{2277:function(e,t,a){a.r(t),a.d(t,{_frontmatter:function(){return l},default:function(){return m}});var n=a(3366),r=(a(7294),a(4983)),i=a(874),o=["components"],l={},s={_frontmatter:l},c=i.Z;function m(e){var t=e.components,a=(0,n.Z)(e,o);return(0,r.kt)(c,Object.assign({},s,a,{components:t,mdxType:"MDXLayout"}),(0,r.kt)("p",null,(0,r.kt)("a",{parentName:"p",href:"https://app.travis-ci.com/IBM/LNN"},(0,r.kt)("img",{parentName:"a",src:"https://app.travis-ci.com/IBM/LNN.svg?branch=master",alt:"Build Status"})),"\n",(0,r.kt)("a",{parentName:"p",href:"https://github.com/IBM/LNN/blob/master/LICENSE"},(0,r.kt)("img",{parentName:"a",src:"https://img.shields.io/github/license/IBM/LNN",alt:"License"})),"\n",(0,r.kt)("a",{parentName:"p",href:"https://github.com/psf/black"},(0,r.kt)("img",{parentName:"a",src:"https://img.shields.io/badge/code%20style-black-000000.svg",alt:"Code style: Black"}))),(0,r.kt)("h1",null,"Logical Neural Networks"),(0,r.kt)("p",null,"LNNs are a novel ",(0,r.kt)("inlineCode",{parentName:"p"},"Neuro = Symbolic")," framework designed to seamlessly provide key\nproperties of both neural nets (learning) and symbolic logic (knowledge and reasoning)."),(0,r.kt)("ul",null,(0,r.kt)("li",{parentName:"ul"},"Every neuron has a meaning as a component of a formula in a weighted real-valued logic, yielding a highly interpretable disentangled representation. "),(0,r.kt)("li",{parentName:"ul"},"Inference is omnidirectional, which corresponds to logical reasoning and includes classical first-order logic (FOL) theorem proving with quantifiers."),(0,r.kt)("li",{parentName:"ul"},"Intermediate neurons are inspectable and may individually specify targets and error signals, rather than focussing on predefined target variables at the last layer of the network."),(0,r.kt)("li",{parentName:"ul"},"The model is end-to-end differentiable, and learning minimizes a novel loss function capturing logical contradiction, yielding resilience to inconsistent knowledge. "),(0,r.kt)("li",{parentName:"ul"},"Bounds on truths (as ranges) enable the open-world assumption, which can have probabilistic semantics and offer resilience to incomplete knowledge.")),(0,r.kt)("h2",null,"Quickstart"),(0,r.kt)("p",null,"To install the LNN:"),(0,r.kt)("ol",null,(0,r.kt)("li",{parentName:"ol"},"Install ",(0,r.kt)("a",{parentName:"li",href:"https://www.graphviz.org/download/"},"GraphViz")),(0,r.kt)("li",{parentName:"ol"},"Run: ",(0,r.kt)("pre",{parentName:"li"},(0,r.kt)("code",{parentName:"pre"},"pip install git+https://github.com/IBM/LNN.git\n")))),(0,r.kt)("h2",null,"Documentation"),(0,r.kt)("table",null,(0,r.kt)("thead",{parentName:"table"},(0,r.kt)("tr",{parentName:"thead"},(0,r.kt)("th",{parentName:"tr",align:"center"},(0,r.kt)("a",{parentName:"th",href:"https://ibm.github.io/LNN/"},"Read the Docs")),(0,r.kt)("th",{parentName:"tr",align:"center"},(0,r.kt)("a",{parentName:"th",href:"https://ibm.github.io/LNN/papers.html"},"Academic Papers")),(0,r.kt)("th",{parentName:"tr",align:"center"},(0,r.kt)("a",{parentName:"th",href:"https://ibm.github.io/LNN/education/education.html"},"Educational Resources")),(0,r.kt)("th",{parentName:"tr",align:"center"},(0,r.kt)("a",{parentName:"th",href:"https://research.ibm.com/teams/neuro-symbolic-ai"},"Neuro-Symbolic AI")),(0,r.kt)("th",{parentName:"tr",align:"center"},(0,r.kt)("a",{parentName:"th",href:"https://ibm.github.io/LNN/usage.html"},"API Overview")),(0,r.kt)("th",{parentName:"tr",align:"center"},(0,r.kt)("a",{parentName:"th",href:"https://ibm.github.io/LNN/lnn/LNN.html"},"Python Module")))),(0,r.kt)("tbody",{parentName:"table"},(0,r.kt)("tr",{parentName:"tbody"},(0,r.kt)("td",{parentName:"tr",align:"center"},(0,r.kt)("a",{parentName:"td",href:"https://ibm.github.io/LNN/"},(0,r.kt)("img",{parentName:"a",src:"https://raw.githubusercontent.com/IBM/LNN/master/docsrc/images/icons/doc.png",alt:"Docs",width:60}))),(0,r.kt)("td",{parentName:"tr",align:"center"},(0,r.kt)("a",{parentName:"td",href:"https://ibm.github.io/LNN/papers.html"},(0,r.kt)("img",{parentName:"a",src:"https://raw.githubusercontent.com/IBM/LNN/master/docsrc/images/icons/academic.png",alt:"Academic Papers",width:60}))),(0,r.kt)("td",{parentName:"tr",align:"center"},(0,r.kt)("a",{parentName:"td",href:"https://ibm.github.io/LNN/education/education.html"},(0,r.kt)("img",{parentName:"a",src:"https://raw.githubusercontent.com/IBM/LNN/master/docsrc/images/icons/help.png",alt:"Getting Started",width:60}))),(0,r.kt)("td",{parentName:"tr",align:"center"},(0,r.kt)("a",{parentName:"td",href:"https://research.ibm.com/teams/neuro-symbolic-ai"},(0,r.kt)("img",{parentName:"a",src:"https://raw.githubusercontent.com/IBM/LNN/master/docsrc/images/icons/nsai.png",alt:"Neuro-Symbolic AI",width:60}))),(0,r.kt)("td",{parentName:"tr",align:"center"},(0,r.kt)("a",{parentName:"td",href:"https://ibm.github.io/LNN/usage.html"},(0,r.kt)("img",{parentName:"a",src:"https://raw.githubusercontent.com/IBM/LNN/master/docsrc/images/icons/api.png",alt:"API",width:60}))),(0,r.kt)("td",{parentName:"tr",align:"center"},(0,r.kt)("a",{parentName:"td",href:"https://ibm.github.io/LNN/lnn/LNN.html"},(0,r.kt)("img",{parentName:"a",src:"https://raw.githubusercontent.com/IBM/LNN/master/docsrc/images/icons/python.png",alt:"Python Module",width:60})))))),(0,r.kt)("h2",null,"Citation"),(0,r.kt)("p",null,"If you use Logical Neural Networks for research, please consider citing the\nreference paper:"),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre",className:"language-raw"},"@article{riegel2020logical,\n  title={Logical neural networks},\n  author={Riegel, Ryan and Gray, Alexander and Luus, Francois and Khan, Naweed and Makondo, Ndivhuwo and Akhalwaya, Ismail Yunus and Qian, Haifeng and Fagin, Ronald and Barahona, Francisco and Sharma, Udit and others},\n  journal={arXiv preprint arXiv:2006.13155},\n  year={2020}\n}\n")),(0,r.kt)("h2",null,"Main Contributors"),(0,r.kt)("p",null,(0,r.kt)("a",{parentName:"p",href:"https://github.com/NaweedAghmad"},"Naweed Khan"),", ",(0,r.kt)("a",{parentName:"p",href:"https://github.com/nDiv"},"Ndivhuwo Makondo"),", ",(0,r.kt)("a",{parentName:"p",href:"https://github.com/francoisluus"},"Francois Luus"),", Dheeraj Sreedhar, Ismail Akhalwaya, ",(0,r.kt)("a",{parentName:"p",href:"https://github.com/richardyoung00"},"Richard Young"),", ",(0,r.kt)("a",{parentName:"p",href:"https://github.com/tobykurien"},"Toby Kurien")))}m.isMDXComponent=!0},6156:function(e,t,a){a.d(t,{Z:function(){return i}});var n=a(7294),r=a(36),i=function(e){var t=e.date,a=new Date(t);return t?n.createElement(r.X2,{className:"last-modified-date-module--row--XJoYQ"},n.createElement(r.sg,null,n.createElement("div",{className:"last-modified-date-module--text--ogPQF"},"Page last updated: ",a.toLocaleDateString("en-GB",{day:"2-digit",year:"numeric",month:"long"})))):null}},7574:function(e,t,a){var n=a(7294),r=a(5444),i=a(6258),o=a(2565);function l(e,t){var a="undefined"!=typeof Symbol&&e[Symbol.iterator]||e["@@iterator"];if(a)return(a=a.call(e)).next.bind(a);if(Array.isArray(e)||(a=function(e,t){if(!e)return;if("string"==typeof e)return s(e,t);var a=Object.prototype.toString.call(e).slice(8,-1);"Object"===a&&e.constructor&&(a=e.constructor.name);if("Map"===a||"Set"===a)return Array.from(e);if("Arguments"===a||/^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(a))return s(e,t)}(e))||t&&e&&"number"==typeof e.length){a&&(e=a);var n=0;return function(){return n>=e.length?{done:!0}:{done:!1,value:e[n++]}}}throw new TypeError("Invalid attempt to iterate non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.")}function s(e,t){(null==t||t>e.length)&&(t=e.length);for(var a=0,n=new Array(t);a<t;a++)n[a]=e[a];return n}var c=function(e){return o.find((function(t){return t.key===e}))||!1},m=function(e,t){var a=function(e,t){var a=t.replace("/repos/","");return e.allMdx.edges.filter((function(e){return e.node.slug===a}))[0].node}(e,t),o=a.frontmatter,s="/repos/"+a.slug,m=n.createElement("div",null,n.createElement("div",{className:i.pb},n.createElement("h4",null,o.title),n.createElement("p",{className:i.pU},o.description)),n.createElement("p",{className:i.pt},function(e){for(var t,a=[],r=l(e);!(t=r()).done;){var i=t.value;a.push(n.createElement("button",{class:"bx--tag bx--tag--green"}," ",n.createElement("span",{class:"bx--tag__label",title:c(i).name},i)," "))}return a}(o.tags)));return n.createElement(r.Link,{to:s,className:i.Gg},m)};t.Z=function(e){return n.createElement(r.StaticQuery,{query:"3281138953",render:function(t){return m(t,e.to)}})}},9195:function(e,t,a){var n=a(7294),r=a(6258);t.Z=function(e){return n.createElement("div",{className:r.fU},e.children)}},874:function(e,t,a){a.d(t,{Z:function(){return L}});var n=a(7294),r=a(8650),i=a.n(r),o=a(5444),l=a(4983),s=a(5426),c=a(4311),m=a(808),u=a(8318),p=a(4275),h=a(9851),d=a(2881),g=a(6958),N=a(6156),f=a(2565);function b(e,t){var a="undefined"!=typeof Symbol&&e[Symbol.iterator]||e["@@iterator"];if(a)return(a=a.call(e)).next.bind(a);if(Array.isArray(e)||(a=function(e,t){if(!e)return;if("string"==typeof e)return k(e,t);var a=Object.prototype.toString.call(e).slice(8,-1);"Object"===a&&e.constructor&&(a=e.constructor.name);if("Map"===a||"Set"===a)return Array.from(e);if("Arguments"===a||/^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(a))return k(e,t)}(e))||t&&e&&"number"==typeof e.length){a&&(e=a);var n=0;return function(){return n>=e.length?{done:!0}:{done:!1,value:e[n++]}}}throw new TypeError("Invalid attempt to iterate non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.")}function k(e,t){(null==t||t>e.length)&&(t=e.length);for(var a=0,n=new Array(t);a<t;a++)n[a]=e[a];return n}var y=function(e){return f.find((function(t){return t.key===e}))||!1},v=function(e){for(var t,a=e.frontmatter,r=a.url,i=[],l=b(a.tags.entries());!(t=l()).done;){var s=t.value,c=s[0],m=s[1];i.push(n.createElement(o.Link,{to:"/repos#"+m,key:c},n.createElement("button",{class:"bx--tag bx--tag--green"}," ",n.createElement("span",{class:"bx--tag__label",title:y(m).name},m)," ")))}return n.createElement("div",{className:"bx--grid"},n.createElement("div",{className:"bx--row"},n.createElement("div",{className:"bx--col-lg-1"},"Repository: "),n.createElement("div",{className:"bx--col-lg-4"},n.createElement("a",{href:r,target:"_blank",rel:"noreferrer"},r))),n.createElement("div",{className:"bx--row"},n.createElement("div",{className:"bx--col-lg-1 category-header"},"Categories:"),n.createElement("div",{className:"bx--col-lg-4"},n.createElement("div",{className:"RepoHeader-module--flex_sm--FX8Eh"},i))))},w=a(7574),E=a(9195),L=function(e){var t=e.pageContext,a=e.children,r=e.location,f=e.Title,b=t.frontmatter,k=void 0===b?{}:b,y=t.relativePagePath,L=t.titleType,I=k.tabs,x=k.title,A=k.theme,M=k.description,S=k.keywords,Z=k.date,B=(0,g.Z)().interiorTheme,_={RepoLink:w.Z,RepoLinkList:E.Z,Link:o.Link},C=(0,o.useStaticQuery)("2102389209").site.pathPrefix,P=C?r.pathname.replace(C,""):r.pathname,T=I?P.split("/").filter(Boolean).slice(-1)[0]||i()(I[0],{lower:!0}):"",R=A||B;return n.createElement(c.Z,{tabs:I,homepage:!1,theme:R,pageTitle:x,pageDescription:M,pageKeywords:S,titleType:L},n.createElement(m.Z,{title:f?n.createElement(f,null):x,label:"label",tabs:I,theme:R}),I&&n.createElement(h.Z,{title:x,slug:P,tabs:I,currentTab:T}),n.createElement(d.Z,{padded:!0},n.createElement(v,{frontmatter:k}),n.createElement(l.Zo,{components:_},a),n.createElement(u.Z,{relativePagePath:y}),n.createElement(N.Z,{date:Z})),n.createElement(p.Z,{pageContext:t,location:r,slug:P,tabs:I,currentTab:T}),n.createElement(s.Z,null))}}}]);
//# sourceMappingURL=component---src-pages-repos-lnn-mdx-721760f2059fad25e941.js.map