"use strict";(self.webpackChunkneuro_symbolic_ai_toolkit_site=self.webpackChunkneuro_symbolic_ai_toolkit_site||[]).push([[3651],{2458:function(e,t,r){r.r(t),r.d(t,{_frontmatter:function(){return l},default:function(){return u}});var n=r(3366),a=(r(7294),r(4983)),o=r(874),i=["components"],l={},s={_frontmatter:l},c=o.Z;function u(e){var t=e.components,r=(0,n.Z)(e,i);return(0,a.kt)(c,Object.assign({},s,r,{components:t,mdxType:"MDXLayout"}),(0,a.kt)("h2",null,"Description"),(0,a.kt)("p",null,"Open source package for accelerated symbolic discovery of fundamental laws, the system integrates data-driven symbolic regression with knowledge-based automated reasoning machinary. The repo include 3 main components:"),(0,a.kt)("ol",null,(0,a.kt)("li",{parentName:"ol"},"symbolic-regression - symbolic regression module that offers various methods for formation of symbolic hypotheses based on given data, super-set of permissible operators, and other desired prefrences"),(0,a.kt)("li",{parentName:"ol"},"symbolic-discovery-reasoning - differential dynamic reasoning module that recieves background theory and set of hypotheses and qualifies as for their degree of derivability"),(0,a.kt)("li",{parentName:"ol"},"experimental-design - given a set of cadidate hypotheses, of various funcitonal forms and / or parameterization, proposes experiments to establish which hypothesis is more likely ")),(0,a.kt)("p",null,"For more details please refer to the following mansucirpts:"),(0,a.kt)("ul",null,(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("a",{parentName:"li",href:"https://arxiv.org/abs/2109.01634"},"https://arxiv.org/abs/2109.01634")," AI Descartes: Combining Data and Theory for Derivable Scientific Discovery, C Cornelio, S Dash, V Austel, T Josephson, J Goncalves, K Clarkson, N Megiddo, B El Khadir, L Horesh"),(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("a",{parentName:"li",href:"https://arxiv.org/abs/2006.06813"},"https://arxiv.org/abs/2006.06813")," Symbolic Regression using Mixed-Integer Nonlinear Optimization, V Austel, C Cornelio, S Dash, J Goncalves, L Horesh, T Josephson, N Megiddo")),(0,a.kt)("h2",null,"Main Contributors"),(0,a.kt)("p",null,"Sanjeeb Dash, Joao Goncalves, Cristina Cornelio, Ken Clarkson, Lior Horesh"))}u.isMDXComponent=!0},6156:function(e,t,r){r.d(t,{Z:function(){return o}});var n=r(7294),a=r(36),o=function(e){var t=e.date,r=new Date(t);return t?n.createElement(a.X2,{className:"last-modified-date-module--row--XJoYQ"},n.createElement(a.sg,null,n.createElement("div",{className:"last-modified-date-module--text--ogPQF"},"Page last updated: ",r.toLocaleDateString("en-GB",{day:"2-digit",year:"numeric",month:"long"})))):null}},7574:function(e,t,r){var n=r(7294),a=r(5444),o=r(6258),i=r(2565);function l(e,t){var r="undefined"!=typeof Symbol&&e[Symbol.iterator]||e["@@iterator"];if(r)return(r=r.call(e)).next.bind(r);if(Array.isArray(e)||(r=function(e,t){if(!e)return;if("string"==typeof e)return s(e,t);var r=Object.prototype.toString.call(e).slice(8,-1);"Object"===r&&e.constructor&&(r=e.constructor.name);if("Map"===r||"Set"===r)return Array.from(e);if("Arguments"===r||/^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(r))return s(e,t)}(e))||t&&e&&"number"==typeof e.length){r&&(e=r);var n=0;return function(){return n>=e.length?{done:!0}:{done:!1,value:e[n++]}}}throw new TypeError("Invalid attempt to iterate non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.")}function s(e,t){(null==t||t>e.length)&&(t=e.length);for(var r=0,n=new Array(t);r<t;r++)n[r]=e[r];return n}var c=function(e){return i.find((function(t){return t.key===e}))||!1},u=function(e,t){var r=function(e,t){var r=t.replace("/repos/","");return e.allMdx.edges.filter((function(e){return e.node.slug===r}))[0].node}(e,t),i=r.frontmatter,s="/repos/"+r.slug,u=n.createElement("div",null,n.createElement("div",{className:o.pb},n.createElement("h4",null,i.title),n.createElement("p",{className:o.pU},i.description)),n.createElement("p",{className:o.pt},function(e){for(var t,r=[],a=l(e);!(t=a()).done;){var o=t.value;r.push(n.createElement("button",{class:"bx--tag bx--tag--green"}," ",n.createElement("span",{class:"bx--tag__label",title:c(o).name},o)," "))}return r}(i.tags)));return n.createElement(a.Link,{to:s,className:o.Gg},u)};t.Z=function(e){return n.createElement(a.StaticQuery,{query:"3281138953",render:function(t){return u(t,e.to)}})}},9195:function(e,t,r){var n=r(7294),a=r(6258);t.Z=function(e){return n.createElement("div",{className:a.fU},e.children)}},874:function(e,t,r){r.d(t,{Z:function(){return w}});var n=r(7294),a=r(8650),o=r.n(a),i=r(5444),l=r(4983),s=r(5426),c=r(4311),u=r(808),m=r(8318),d=r(4275),f=r(9851),p=r(2881),g=r(6958),b=r(6156),h=r(2565);function y(e,t){var r="undefined"!=typeof Symbol&&e[Symbol.iterator]||e["@@iterator"];if(r)return(r=r.call(e)).next.bind(r);if(Array.isArray(e)||(r=function(e,t){if(!e)return;if("string"==typeof e)return v(e,t);var r=Object.prototype.toString.call(e).slice(8,-1);"Object"===r&&e.constructor&&(r=e.constructor.name);if("Map"===r||"Set"===r)return Array.from(e);if("Arguments"===r||/^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(r))return v(e,t)}(e))||t&&e&&"number"==typeof e.length){r&&(e=r);var n=0;return function(){return n>=e.length?{done:!0}:{done:!1,value:e[n++]}}}throw new TypeError("Invalid attempt to iterate non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.")}function v(e,t){(null==t||t>e.length)&&(t=e.length);for(var r=0,n=new Array(t);r<t;r++)n[r]=e[r];return n}var E=function(e){return h.find((function(t){return t.key===e}))||!1},k=function(e){for(var t,r=e.frontmatter,a=r.url,o=[],l=y(r.tags.entries());!(t=l()).done;){var s=t.value,c=s[0],u=s[1];o.push(n.createElement(i.Link,{to:"/repos#"+u,key:c},n.createElement("button",{class:"bx--tag bx--tag--green"}," ",n.createElement("span",{class:"bx--tag__label",title:E(u).name},u)," ")))}return n.createElement("div",{className:"bx--grid"},n.createElement("div",{className:"bx--row"},n.createElement("div",{className:"bx--col-lg-1"},"Repository: "),n.createElement("div",{className:"bx--col-lg-4"},n.createElement("a",{href:a,target:"_blank",rel:"noreferrer"},a))),n.createElement("div",{className:"bx--row"},n.createElement("div",{className:"bx--col-lg-1 category-header"},"Categories:"),n.createElement("div",{className:"bx--col-lg-4"},n.createElement("div",{className:"RepoHeader-module--flex_sm--FX8Eh"},o))))},x=r(7574),N=r(9195),w=function(e){var t=e.pageContext,r=e.children,a=e.location,h=e.Title,y=t.frontmatter,v=void 0===y?{}:y,E=t.relativePagePath,w=t.titleType,C=v.tabs,S=v.title,Z=v.theme,_=v.description,A=v.keywords,T=v.date,D=(0,g.Z)().interiorTheme,L={RepoLink:x.Z,RepoLinkList:N.Z,Link:i.Link},M=(0,i.useStaticQuery)("2102389209").site.pathPrefix,j=M?a.pathname.replace(M,""):a.pathname,I=C?j.split("/").filter(Boolean).slice(-1)[0]||o()(C[0],{lower:!0}):"",O=Z||D;return n.createElement(c.Z,{tabs:C,homepage:!1,theme:O,pageTitle:S,pageDescription:_,pageKeywords:A,titleType:w},n.createElement(u.Z,{title:h?n.createElement(h,null):S,label:"label",tabs:C,theme:O}),C&&n.createElement(f.Z,{title:S,slug:j,tabs:C,currentTab:I}),n.createElement(p.Z,{padded:!0},n.createElement(k,{frontmatter:v}),n.createElement(l.Zo,{components:L},r),n.createElement(m.Z,{relativePagePath:E}),n.createElement(b.Z,{date:T})),n.createElement(d.Z,{pageContext:t,location:a,slug:j,tabs:C,currentTab:I}),n.createElement(s.Z,null))}}}]);
//# sourceMappingURL=component---src-pages-repos-discovery-mdx-de445a1133c1c083d381.js.map