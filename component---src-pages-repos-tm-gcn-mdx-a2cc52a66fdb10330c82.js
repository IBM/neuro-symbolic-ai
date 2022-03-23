"use strict";(self.webpackChunkneuro_symbolic_ai_toolkit_site=self.webpackChunkneuro_symbolic_ai_toolkit_site||[]).push([[8039],{9660:function(e,t,r){r.r(t),r.d(t,{_frontmatter:function(){return i},default:function(){return u}});var n=r(3366),a=(r(7294),r(4983)),o=r(874),l=["components"],i={},c={_frontmatter:i},s=o.Z;function u(e){var t=e.components,r=(0,n.Z)(e,l);return(0,a.kt)(s,Object.assign({},c,r,{components:t,mdxType:"MDXLayout"}),(0,a.kt)("h2",null,"Description"),(0,a.kt)("p",null,"Pytorch code for the TM-GCN method, a Dynamic Graph Convolutional Networks formulated using the Tensor M-Product algebra. "),(0,a.kt)("ul",null,(0,a.kt)("li",{parentName:"ul"},"Tensor-tensor algebra for optimal representation and compression of multiway data, ME Kilmer, L Horesh, H Avron, E Newman, Proceedings of the National Academy of Science (PNAS), 2021, ",(0,a.kt)("a",{parentName:"li",href:"https://www.pnas.org/content/118/28/e2015851118.short"},"https://www.pnas.org/content/118/28/e2015851118.short")),(0,a.kt)("li",{parentName:"ul"},"Dynamic Graph Convolutional Networks using the tensor M-product, O Malik, S Ubaru, L Horesh, M Kilmer, H Avron, SIAM International Conference on Data Mining (SDM 21), 2021, ",(0,a.kt)("a",{parentName:"li",href:"https://epubs.siam.org/doi/abs/10.1137/1.9781611976700.82"},"https://epubs.siam.org/doi/abs/10.1137/1.9781611976700.82")),(0,a.kt)("li",{parentName:"ul"},"New tensor algebra changes the rules of data analysis, IBM Blog, ",(0,a.kt)("a",{parentName:"li",href:"https://research.ibm.com/blog/new-tensor-algebra"},"https://research.ibm.com/blog/new-tensor-algebra"))),(0,a.kt)("h2",null,"Main Contributors"),(0,a.kt)("p",null,"Shashanka Ubaru, Osman Malik, Lior Horesh"))}u.isMDXComponent=!0},6156:function(e,t,r){r.d(t,{Z:function(){return o}});var n=r(7294),a=r(36),o=function(e){var t=e.date,r=new Date(t);return t?n.createElement(a.X2,{className:"last-modified-date-module--row--XJoYQ"},n.createElement(a.sg,null,n.createElement("div",{className:"last-modified-date-module--text--ogPQF"},"Page last updated: ",r.toLocaleDateString("en-GB",{day:"2-digit",year:"numeric",month:"long"})))):null}},7574:function(e,t,r){var n=r(7294),a=r(5444),o=r(6258),l=r(2565);function i(e,t){var r="undefined"!=typeof Symbol&&e[Symbol.iterator]||e["@@iterator"];if(r)return(r=r.call(e)).next.bind(r);if(Array.isArray(e)||(r=function(e,t){if(!e)return;if("string"==typeof e)return c(e,t);var r=Object.prototype.toString.call(e).slice(8,-1);"Object"===r&&e.constructor&&(r=e.constructor.name);if("Map"===r||"Set"===r)return Array.from(e);if("Arguments"===r||/^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(r))return c(e,t)}(e))||t&&e&&"number"==typeof e.length){r&&(e=r);var n=0;return function(){return n>=e.length?{done:!0}:{done:!1,value:e[n++]}}}throw new TypeError("Invalid attempt to iterate non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.")}function c(e,t){(null==t||t>e.length)&&(t=e.length);for(var r=0,n=new Array(t);r<t;r++)n[r]=e[r];return n}var s=function(e){return l.find((function(t){return t.key===e}))||!1},u=function(e,t){var r=function(e,t){var r=t.replace("/repos/","");return e.allMdx.edges.filter((function(e){return e.node.slug===r}))[0].node}(e,t),l=r.frontmatter,c="/repos/"+r.slug,u=n.createElement("div",null,n.createElement("div",{className:o.pb},n.createElement("h4",null,l.title),n.createElement("p",{className:o.pU},l.description)),n.createElement("p",{className:o.pt},function(e){for(var t,r=[],a=i(e);!(t=a()).done;){var o=t.value;r.push(n.createElement("button",{class:"bx--tag bx--tag--green"}," ",n.createElement("span",{class:"bx--tag__label",title:s(o).name},o)," "))}return r}(l.tags)));return n.createElement(a.Link,{to:c,className:o.Gg},u)};t.Z=function(e){return n.createElement(a.StaticQuery,{query:"3281138953",render:function(t){return u(t,e.to)}})}},9195:function(e,t,r){var n=r(7294),a=r(6258);t.Z=function(e){return n.createElement("div",{className:a.fU},e.children)}},874:function(e,t,r){r.d(t,{Z:function(){return x}});var n=r(7294),a=r(8650),o=r.n(a),l=r(5444),i=r(4983),c=r(5426),s=r(4311),u=r(808),m=r(8318),f=r(4275),p=r(9851),d=r(2881),b=r(6958),g=r(6156),h=r(2565);function y(e,t){var r="undefined"!=typeof Symbol&&e[Symbol.iterator]||e["@@iterator"];if(r)return(r=r.call(e)).next.bind(r);if(Array.isArray(e)||(r=function(e,t){if(!e)return;if("string"==typeof e)return v(e,t);var r=Object.prototype.toString.call(e).slice(8,-1);"Object"===r&&e.constructor&&(r=e.constructor.name);if("Map"===r||"Set"===r)return Array.from(e);if("Arguments"===r||/^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(r))return v(e,t)}(e))||t&&e&&"number"==typeof e.length){r&&(e=r);var n=0;return function(){return n>=e.length?{done:!0}:{done:!1,value:e[n++]}}}throw new TypeError("Invalid attempt to iterate non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.")}function v(e,t){(null==t||t>e.length)&&(t=e.length);for(var r=0,n=new Array(t);r<t;r++)n[r]=e[r];return n}var E=function(e){return h.find((function(t){return t.key===e}))||!1},k=function(e){for(var t,r=e.frontmatter,a=r.url,o=[],i=y(r.tags.entries());!(t=i()).done;){var c=t.value,s=c[0],u=c[1];o.push(n.createElement(l.Link,{to:"/repos#"+u,key:s},n.createElement("button",{class:"bx--tag bx--tag--green"}," ",n.createElement("span",{class:"bx--tag__label",title:E(u).name},u)," ")))}return n.createElement("div",{className:"bx--grid"},n.createElement("div",{className:"bx--row"},n.createElement("div",{className:"bx--col-lg-1"},"Repository: "),n.createElement("div",{className:"bx--col-lg-4"},n.createElement("a",{href:a,target:"_blank",rel:"noreferrer"},a))),n.createElement("div",{className:"bx--row"},n.createElement("div",{className:"bx--col-lg-1 category-header"},"Categories:"),n.createElement("div",{className:"bx--col-lg-4"},n.createElement("div",{className:"RepoHeader-module--flex_sm--FX8Eh"},o))))},w=r(7574),N=r(9195),x=function(e){var t=e.pageContext,r=e.children,a=e.location,h=e.Title,y=t.frontmatter,v=void 0===y?{}:y,E=t.relativePagePath,x=t.titleType,S=v.tabs,Z=v.title,A=v.theme,M=v.description,_=v.keywords,C=v.date,T=(0,b.Z)().interiorTheme,L={RepoLink:w.Z,RepoLinkList:N.Z,Link:l.Link},P=(0,l.useStaticQuery)("2102389209").site.pathPrefix,D=P?a.pathname.replace(P,""):a.pathname,I=S?D.split("/").filter(Boolean).slice(-1)[0]||o()(S[0],{lower:!0}):"",j=A||T;return n.createElement(s.Z,{tabs:S,homepage:!1,theme:j,pageTitle:Z,pageDescription:M,pageKeywords:_,titleType:x},n.createElement(u.Z,{title:h?n.createElement(h,null):Z,label:"label",tabs:S,theme:j}),S&&n.createElement(p.Z,{title:Z,slug:D,tabs:S,currentTab:I}),n.createElement(d.Z,{padded:!0},n.createElement(k,{frontmatter:v}),n.createElement(i.Zo,{components:L},r),n.createElement(m.Z,{relativePagePath:E}),n.createElement(g.Z,{date:C})),n.createElement(f.Z,{pageContext:t,location:a,slug:D,tabs:S,currentTab:I}),n.createElement(c.Z,null))}}}]);
//# sourceMappingURL=component---src-pages-repos-tm-gcn-mdx-a2cc52a66fdb10330c82.js.map