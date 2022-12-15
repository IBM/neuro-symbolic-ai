"use strict";(self.webpackChunkneuro_symbolic_ai_toolkit_site=self.webpackChunkneuro_symbolic_ai_toolkit_site||[]).push([[3328],{908:function(e,t,n){n.r(t),n.d(t,{_frontmatter:function(){return l},default:function(){return u}});var r=n(3366),a=(n(7294),n(4983)),o=n(9214),i=["components"],l={},s={_frontmatter:l},c=o.Z;function u(e){var t=e.components,n=(0,r.Z)(e,i);return(0,a.kt)(c,Object.assign({},s,n,{components:t,mdxType:"MDXLayout"}),(0,a.kt)("h2",null,"Description"),(0,a.kt)("p",null,"Open source package for accelerated symbolic discovery of fundamental laws, the system integrates data-driven symbolic regression with knowledge-based automated reasoning machinary. The repo include 3 main components:"),(0,a.kt)("ol",null,(0,a.kt)("li",{parentName:"ol"},"symbolic-regression - symbolic regression module that offers various methods for formation of symbolic hypotheses based on given data, super-set of permissible operators, and other desired prefrences"),(0,a.kt)("li",{parentName:"ol"},"symbolic-discovery-reasoning - differential dynamic reasoning module that recieves background theory and set of hypotheses and qualifies as for their degree of derivability"),(0,a.kt)("li",{parentName:"ol"},"experimental-design - given a set of cadidate hypotheses, of various funcitonal forms and / or parameterization, proposes experiments to establish which hypothesis is more likely ")),(0,a.kt)("p",null,"For more details please refer to the following mansucirpts:"),(0,a.kt)("ul",null,(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("a",{parentName:"li",href:"https://arxiv.org/abs/2109.01634"},"https://arxiv.org/abs/2109.01634")," AI Descartes: Combining Data and Theory for Derivable Scientific Discovery, C Cornelio, S Dash, V Austel, T Josephson, J Goncalves, K Clarkson, N Megiddo, B El Khadir, L Horesh"),(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("a",{parentName:"li",href:"https://arxiv.org/abs/2006.06813"},"https://arxiv.org/abs/2006.06813")," Symbolic Regression using Mixed-Integer Nonlinear Optimization, V Austel, C Cornelio, S Dash, J Goncalves, L Horesh, T Josephson, N Megiddo")),(0,a.kt)("h2",null,"Main Contributors"),(0,a.kt)("p",null,"Sanjeeb Dash, Joao Goncalves, Cristina Cornelio, Ken Clarkson, Lior Horesh"))}u.isMDXComponent=!0},6156:function(e,t,n){n.d(t,{Z:function(){return o}});var r=n(7294),a=n(36),o=function(e){var t=e.date,n=new Date(t);return t?r.createElement(a.X2,{className:"last-modified-date-module--row--XJoYQ"},r.createElement(a.sg,null,r.createElement("div",{className:"last-modified-date-module--text--ogPQF"},"Page last updated: ",n.toLocaleDateString("en-GB",{day:"2-digit",year:"numeric",month:"long"})))):null}},1517:function(e,t,n){var r=n(7294);t.Z=function(e){var t="";return"small"===e.size&&(t="bx--btn--sm"),r.createElement("a",{href:e.href,target:"_blank",rel:"noreferrer",title:e.title,className:"bx--btn bx--btn--primary "+t,style:e.style},e.children)}},1109:function(e,t,n){n.d(t,{Z:function(){return a}});var r=n(7294),a=function(e){return r.createElement("div",{className:"People-module--flex--LApLg"},e.children)}},3167:function(e,t,n){n.d(t,{Z:function(){return a}});var r=n(7294),a=function(e){var t=e.name,n=e.url,a=e.affiliation,o=e.img;return r.createElement("a",{href:n,target:"_blank",rel:"noreferrer",className:"Person-module--link---uaJL"},r.createElement("div",null,r.createElement("img",{src:o,alt:t,title:t,className:"Person-module--roundImg--l8Swm"}),r.createElement("div",{className:"Person-module--textReset--Y0VEx"},r.createElement("div",{className:"Person-module--name--xrah4"},t),r.createElement("div",{className:"Person-module--affiliation--t+v4k"},a))))}},7574:function(e,t,n){var r=n(7294),a=n(5444),o=n(6258),i=n(2565);function l(e,t){var n="undefined"!=typeof Symbol&&e[Symbol.iterator]||e["@@iterator"];if(n)return(n=n.call(e)).next.bind(n);if(Array.isArray(e)||(n=function(e,t){if(!e)return;if("string"==typeof e)return s(e,t);var n=Object.prototype.toString.call(e).slice(8,-1);"Object"===n&&e.constructor&&(n=e.constructor.name);if("Map"===n||"Set"===n)return Array.from(e);if("Arguments"===n||/^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n))return s(e,t)}(e))||t&&e&&"number"==typeof e.length){n&&(e=n);var r=0;return function(){return r>=e.length?{done:!0}:{done:!1,value:e[r++]}}}throw new TypeError("Invalid attempt to iterate non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.")}function s(e,t){(null==t||t>e.length)&&(t=e.length);for(var n=0,r=new Array(t);n<t;n++)r[n]=e[n];return r}var c=function(e){var t=i.findIndex((function(t){return t.key===e}));return-1===t?999:t},u=function(e){return i.find((function(t){return t.key===e}))||!1},m=function(e){for(var t,n=[],a=l(e.sort((function(e,t){return function(e,t){var n=c(e),r=c(t);return n>r?1:n<r?-1:0}(e,t)})));!(t=a()).done;){var o=t.value,i="bx--tag";u(o)?i+=" bx--tag--green":i+=" bx--tag--cool-gray",n.push(r.createElement("button",{class:i}," ",r.createElement("span",{class:"bx--tag__label",title:u(o).name},o)," "))}return n};t.Z=function(e){return r.createElement(a.StaticQuery,{query:"3151510810",render:function(t){return function(e,t){var n=function(e,t){var n=t.replace("/toolkit/","");return e.allMdx.edges.filter((function(e){return e.node.slug===n}))[0].node}(e,t),i=n.frontmatter,l="/toolkit/"+n.slug,s=r.createElement("div",null,r.createElement("div",{className:o.pb},r.createElement("h4",null,i.title),r.createElement("p",{className:o.pU},i.description)),r.createElement("p",{className:o.pt},m(i.tags)));return r.createElement(a.Link,{to:l,className:o.Gg},s)}(t,e.to)}})}},9195:function(e,t,n){var r=n(7294),a=n(6258);t.Z=function(e){return r.createElement("div",{className:a.fU},e.children)}},6179:function(e,t,n){var r=n(7294),a=n(5414),o=n(5444);function i(e){var t=e.description,n=e.lang,i=e.meta,l=e.image,s=e.title,c=e.pathname,u=(0,o.useStaticQuery)("151170173").site,m=t||u.siteMetadata.description,d=null;l&&(d=l.includes("https://")||l.includes("http://")?l:u.siteMetadata.siteUrl+"/"+l);var f=c?""+u.siteMetadata.siteUrl+c:null;return r.createElement(a.q,{htmlAttributes:{lang:n},title:s,titleTemplate:"%s | "+u.siteMetadata.title,link:f?[{rel:"canonical",href:f}]:[],meta:[{name:"description",content:m},{property:"og:title",content:s},{property:"og:description",content:m},{property:"og:type",content:"website"},{name:"twitter:title",content:s},{name:"twitter:description",content:m}].concat(l?[{property:"og:image",content:d},{property:"og:image:width",content:l.width},{property:"og:image:height",content:l.height},{name:"twitter:card",content:"summary_large_image"}]:[{name:"twitter:card",content:"summary"}]).concat(i)})}i.defaultProps={lang:"en",meta:[],description:""},t.Z=i},9214:function(e,t,n){n.d(t,{Z:function(){return A}});var r=n(7294),a=n(8650),o=n.n(a),i=n(5444),l=n(4983),s=n(5426),c=n(8477),u=n(808),m=n(8318),d=n(4275),f=n(9851),p=n(2881),g=n(6958),h=n(6156),b=n(2565);function y(e,t){var n="undefined"!=typeof Symbol&&e[Symbol.iterator]||e["@@iterator"];if(n)return(n=n.call(e)).next.bind(n);if(Array.isArray(e)||(n=function(e,t){if(!e)return;if("string"==typeof e)return v(e,t);var n=Object.prototype.toString.call(e).slice(8,-1);"Object"===n&&e.constructor&&(n=e.constructor.name);if("Map"===n||"Set"===n)return Array.from(e);if("Arguments"===n||/^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n))return v(e,t)}(e))||t&&e&&"number"==typeof e.length){n&&(e=n);var r=0;return function(){return r>=e.length?{done:!0}:{done:!1,value:e[r++]}}}throw new TypeError("Invalid attempt to iterate non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.")}function v(e,t){(null==t||t>e.length)&&(t=e.length);for(var n=0,r=new Array(t);n<t;n++)r[n]=e[n];return r}var E=function(e){var t=b.findIndex((function(t){return t.key===e}));return-1===t?999:t},k=function(e){return b.find((function(t){return t.key===e}))||!1},x=function(e){for(var t,n=e.frontmatter,a=n.url,o=[],l=y(n.tags.sort((function(e,t){return function(e,t){var n=E(e),r=E(t);return n>r?1:n<r?-1:0}(e,t)})));!(t=l()).done;){var s=t.value,c="bx--tag";k(s)?c+=" bx--tag--green":c+=" bx--tag--cool-gray",o.push(r.createElement(i.Link,{to:"/toolkit#"+s,key:s},r.createElement("button",{class:c}," ",r.createElement("span",{class:"bx--tag__label",title:k(s).name},s)," ")))}return r.createElement("div",{className:"bx--grid"},r.createElement("div",{className:"bx--row"},r.createElement("div",{className:"bx--col-lg-1"},"Repository: "),r.createElement("div",{className:"bx--col-lg-4"},r.createElement("a",{href:a,target:"_blank",rel:"noreferrer"},a))),r.createElement("div",{className:"bx--row"},r.createElement("div",{className:"bx--col-lg-1 category-header"},"Categories:"),r.createElement("div",{className:"bx--col-lg-4"},r.createElement("div",{className:"RepoHeader-module--flex_sm--FX8Eh"},o))))},N=n(7574),w=n(9195),Z=n(1517),S=n(3167),_=n(1109),C=n(6179),A=function(e){var t=e.pageContext,n=e.children,a=e.location,b=e.Title,y=t.frontmatter,v=void 0===y?{}:y,E=t.relativePagePath,k=t.titleType,A=v.tabs,L=v.title,P=v.theme,T=v.description,D=v.keywords,M=v.date,I=(0,g.Z)().interiorTheme,j={RepoLink:N.Z,RepoLinkList:w.Z,Link:i.Link,ButtonLink:Z.Z,Person:S.Z,People:_.Z,Seo:C.Z},J=(0,i.useStaticQuery)("2102389209").site.pathPrefix,O=J?a.pathname.replace(J,""):a.pathname,R=A?O.split("/").filter(Boolean).slice(-1)[0]||o()(A[0],{lower:!0}):"",U=P||I;return r.createElement(c.Z,{tabs:A,homepage:!1,theme:U,pageTitle:L,pageDescription:T,pageKeywords:D,titleType:k},r.createElement(u.Z,{title:b?r.createElement(b,null):L,label:"label",tabs:A,theme:U}),A&&r.createElement(f.Z,{title:L,slug:O,tabs:A,currentTab:R}),r.createElement(p.Z,{padded:!0},r.createElement(x,{frontmatter:v}),r.createElement(l.Zo,{components:j},n),r.createElement(m.Z,{relativePagePath:E}),r.createElement(h.Z,{date:M})),r.createElement(d.Z,{pageContext:t,location:a,slug:O,tabs:A,currentTab:R}),r.createElement(s.Z,null))}}}]);
//# sourceMappingURL=component---src-pages-toolkit-ai-descartes-mdx-69bc6ae40b9bfef68c54.js.map