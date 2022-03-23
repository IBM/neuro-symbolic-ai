"use strict";(self.webpackChunkneuro_symbolic_ai_toolkit_site=self.webpackChunkneuro_symbolic_ai_toolkit_site||[]).push([[5483],{1418:function(e,t,r){r.r(t),r.d(t,{_frontmatter:function(){return i},default:function(){return s}});var n=r(3366),a=(r(7294),r(4983)),l=r(874),o=["components"],i={},c={_frontmatter:i},u=l.Z;function s(e){var t=e.components,r=(0,n.Z)(e,o);return(0,a.kt)(u,Object.assign({},c,r,{components:t,mdxType:"MDXLayout"}),(0,a.kt)("h2",null,"Description"),(0,a.kt)("p",null,"This is the HCI component of NeSA. It allows the user to visualize the logical facts, learned policy, accuracy and other metrics. In the future, this will also allow the user to edit the knowledge and the learned policy. It also supports a general purpose visualization and editing tool for any LNN based network."),(0,a.kt)("h2",null,"Main Contributors"),(0,a.kt)("p",null,"Daiki Kimura, Steve Carrow, Stefan Zecevic"))}s.isMDXComponent=!0},6156:function(e,t,r){r.d(t,{Z:function(){return l}});var n=r(7294),a=r(36),l=function(e){var t=e.date,r=new Date(t);return t?n.createElement(a.X2,{className:"last-modified-date-module--row--XJoYQ"},n.createElement(a.sg,null,n.createElement("div",{className:"last-modified-date-module--text--ogPQF"},"Page last updated: ",r.toLocaleDateString("en-GB",{day:"2-digit",year:"numeric",month:"long"})))):null}},7574:function(e,t,r){var n=r(7294),a=r(5444),l=r(6258),o=r(2565);function i(e,t){var r="undefined"!=typeof Symbol&&e[Symbol.iterator]||e["@@iterator"];if(r)return(r=r.call(e)).next.bind(r);if(Array.isArray(e)||(r=function(e,t){if(!e)return;if("string"==typeof e)return c(e,t);var r=Object.prototype.toString.call(e).slice(8,-1);"Object"===r&&e.constructor&&(r=e.constructor.name);if("Map"===r||"Set"===r)return Array.from(e);if("Arguments"===r||/^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(r))return c(e,t)}(e))||t&&e&&"number"==typeof e.length){r&&(e=r);var n=0;return function(){return n>=e.length?{done:!0}:{done:!1,value:e[n++]}}}throw new TypeError("Invalid attempt to iterate non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.")}function c(e,t){(null==t||t>e.length)&&(t=e.length);for(var r=0,n=new Array(t);r<t;r++)n[r]=e[r];return n}var u=function(e){return o.find((function(t){return t.key===e}))||!1},s=function(e,t){var r=function(e,t){var r=t.replace("/repos/","");return e.allMdx.edges.filter((function(e){return e.node.slug===r}))[0].node}(e,t),o=r.frontmatter,c="/repos/"+r.slug,s=n.createElement("div",null,n.createElement("div",{className:l.pb},n.createElement("h4",null,o.title),n.createElement("p",{className:l.pU},o.description)),n.createElement("p",{className:l.pt},function(e){for(var t,r=[],a=i(e);!(t=a()).done;){var l=t.value;r.push(n.createElement("button",{class:"bx--tag bx--tag--green"}," ",n.createElement("span",{class:"bx--tag__label",title:u(l).name},l)," "))}return r}(o.tags)));return n.createElement(a.Link,{to:c,className:l.Gg},s)};t.Z=function(e){return n.createElement(a.StaticQuery,{query:"3281138953",render:function(t){return s(t,e.to)}})}},9195:function(e,t,r){var n=r(7294),a=r(6258);t.Z=function(e){return n.createElement("div",{className:a.fU},e.children)}},874:function(e,t,r){r.d(t,{Z:function(){return Z}});var n=r(7294),a=r(8650),l=r.n(a),o=r(5444),i=r(4983),c=r(5426),u=r(4311),s=r(808),m=r(8318),f=r(4275),d=r(9851),p=r(2881),b=r(6958),g=r(6156),v=r(2565);function y(e,t){var r="undefined"!=typeof Symbol&&e[Symbol.iterator]||e["@@iterator"];if(r)return(r=r.call(e)).next.bind(r);if(Array.isArray(e)||(r=function(e,t){if(!e)return;if("string"==typeof e)return h(e,t);var r=Object.prototype.toString.call(e).slice(8,-1);"Object"===r&&e.constructor&&(r=e.constructor.name);if("Map"===r||"Set"===r)return Array.from(e);if("Arguments"===r||/^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(r))return h(e,t)}(e))||t&&e&&"number"==typeof e.length){r&&(e=r);var n=0;return function(){return n>=e.length?{done:!0}:{done:!1,value:e[n++]}}}throw new TypeError("Invalid attempt to iterate non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.")}function h(e,t){(null==t||t>e.length)&&(t=e.length);for(var r=0,n=new Array(t);r<t;r++)n[r]=e[r];return n}var E=function(e){return v.find((function(t){return t.key===e}))||!1},k=function(e){for(var t,r=e.frontmatter,a=r.url,l=[],i=y(r.tags.entries());!(t=i()).done;){var c=t.value,u=c[0],s=c[1];l.push(n.createElement(o.Link,{to:"/repos#"+s,key:u},n.createElement("button",{class:"bx--tag bx--tag--green"}," ",n.createElement("span",{class:"bx--tag__label",title:E(s).name},s)," ")))}return n.createElement("div",{className:"bx--grid"},n.createElement("div",{className:"bx--row"},n.createElement("div",{className:"bx--col-lg-1"},"Repository: "),n.createElement("div",{className:"bx--col-lg-4"},n.createElement("a",{href:a,target:"_blank",rel:"noreferrer"},a))),n.createElement("div",{className:"bx--row"},n.createElement("div",{className:"bx--col-lg-1 category-header"},"Categories:"),n.createElement("div",{className:"bx--col-lg-4"},n.createElement("div",{className:"RepoHeader-module--flex_sm--FX8Eh"},l))))},x=r(7574),w=r(9195),Z=function(e){var t=e.pageContext,r=e.children,a=e.location,v=e.Title,y=t.frontmatter,h=void 0===y?{}:y,E=t.relativePagePath,Z=t.titleType,N=h.tabs,S=h.title,_=h.theme,A=h.description,C=h.keywords,T=h.date,I=(0,b.Z)().interiorTheme,L={RepoLink:x.Z,RepoLinkList:w.Z,Link:o.Link},j=(0,o.useStaticQuery)("2102389209").site.pathPrefix,D=j?a.pathname.replace(j,""):a.pathname,P=N?D.split("/").filter(Boolean).slice(-1)[0]||l()(N[0],{lower:!0}):"",M=_||I;return n.createElement(u.Z,{tabs:N,homepage:!1,theme:M,pageTitle:S,pageDescription:A,pageKeywords:C,titleType:Z},n.createElement(s.Z,{title:v?n.createElement(v,null):S,label:"label",tabs:N,theme:M}),N&&n.createElement(d.Z,{title:S,slug:D,tabs:N,currentTab:P}),n.createElement(p.Z,{padded:!0},n.createElement(k,{frontmatter:h}),n.createElement(i.Zo,{components:L},r),n.createElement(m.Z,{relativePagePath:E}),n.createElement(g.Z,{date:T})),n.createElement(f.Z,{pageContext:t,location:a,slug:D,tabs:N,currentTab:P}),n.createElement(c.Z,null))}}}]);
//# sourceMappingURL=component---src-pages-repos-nesa-demo-mdx-3add3ac0cd04322e6e89.js.map