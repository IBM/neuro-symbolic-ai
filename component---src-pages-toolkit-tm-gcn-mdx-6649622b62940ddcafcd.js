"use strict";(self.webpackChunkneuro_symbolic_ai_toolkit_site=self.webpackChunkneuro_symbolic_ai_toolkit_site||[]).push([[7181],{1774:function(e,t,n){n.r(t),n.d(t,{_frontmatter:function(){return i},default:function(){return u}});var r=n(3366),a=(n(7294),n(4983)),o=n(9214),l=["components"],i={},c={_frontmatter:i},s=o.Z;function u(e){var t=e.components,n=(0,r.Z)(e,l);return(0,a.kt)(s,Object.assign({},c,n,{components:t,mdxType:"MDXLayout"}),(0,a.kt)("h2",null,"Description"),(0,a.kt)("p",null,"Pytorch code for the TM-GCN method, a Dynamic Graph Convolutional Networks formulated using the Tensor M-Product algebra. "),(0,a.kt)("ul",null,(0,a.kt)("li",{parentName:"ul"},"Tensor-tensor algebra for optimal representation and compression of multiway data, ME Kilmer, L Horesh, H Avron, E Newman, Proceedings of the National Academy of Science (PNAS), 2021, ",(0,a.kt)("a",{parentName:"li",href:"https://www.pnas.org/content/118/28/e2015851118.short"},"https://www.pnas.org/content/118/28/e2015851118.short")),(0,a.kt)("li",{parentName:"ul"},"Dynamic Graph Convolutional Networks using the tensor M-product, O Malik, S Ubaru, L Horesh, M Kilmer, H Avron, SIAM International Conference on Data Mining (SDM 21), 2021, ",(0,a.kt)("a",{parentName:"li",href:"https://epubs.siam.org/doi/abs/10.1137/1.9781611976700.82"},"https://epubs.siam.org/doi/abs/10.1137/1.9781611976700.82")),(0,a.kt)("li",{parentName:"ul"},"New tensor algebra changes the rules of data analysis, IBM Blog, ",(0,a.kt)("a",{parentName:"li",href:"https://research.ibm.com/blog/new-tensor-algebra"},"https://research.ibm.com/blog/new-tensor-algebra"))),(0,a.kt)("h2",null,"Main Contributors"),(0,a.kt)("p",null,"Shashanka Ubaru, Osman Malik, Lior Horesh"))}u.isMDXComponent=!0},6156:function(e,t,n){n.d(t,{Z:function(){return o}});var r=n(7294),a=n(36),o=function(e){var t=e.date,n=new Date(t);return t?r.createElement(a.X2,{className:"last-modified-date-module--row--XJoYQ"},r.createElement(a.sg,null,r.createElement("div",{className:"last-modified-date-module--text--ogPQF"},"Page last updated: ",n.toLocaleDateString("en-GB",{day:"2-digit",year:"numeric",month:"long"})))):null}},1517:function(e,t,n){var r=n(7294);t.Z=function(e){return r.createElement("a",{href:e.href,target:"_blank",rel:"noreferrer",title:e.title,className:"bx--btn bx--btn--primary"},e.children)}},1109:function(e,t,n){n.d(t,{Z:function(){return a}});var r=n(7294),a=function(e){return r.createElement("div",{className:"People-module--flex--LApLg"},e.children)}},3167:function(e,t,n){n.d(t,{Z:function(){return a}});var r=n(7294),a=function(e){var t=e.name,n=e.url,a=e.affiliation,o=e.img;return r.createElement("a",{href:n,target:"_blank",rel:"noreferrer",className:"Person-module--link---uaJL"},r.createElement("div",null,r.createElement("img",{src:o,alt:t,title:t,className:"Person-module--roundImg--l8Swm"}),r.createElement("div",{className:"Person-module--textReset--Y0VEx"},r.createElement("div",{className:"Person-module--name--xrah4"},t),r.createElement("div",{className:"Person-module--affiliation--t+v4k"},a))))}},7574:function(e,t,n){var r=n(7294),a=n(5444),o=n(6258),l=n(2565);function i(e,t){var n="undefined"!=typeof Symbol&&e[Symbol.iterator]||e["@@iterator"];if(n)return(n=n.call(e)).next.bind(n);if(Array.isArray(e)||(n=function(e,t){if(!e)return;if("string"==typeof e)return c(e,t);var n=Object.prototype.toString.call(e).slice(8,-1);"Object"===n&&e.constructor&&(n=e.constructor.name);if("Map"===n||"Set"===n)return Array.from(e);if("Arguments"===n||/^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n))return c(e,t)}(e))||t&&e&&"number"==typeof e.length){n&&(e=n);var r=0;return function(){return r>=e.length?{done:!0}:{done:!1,value:e[r++]}}}throw new TypeError("Invalid attempt to iterate non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.")}function c(e,t){(null==t||t>e.length)&&(t=e.length);for(var n=0,r=new Array(t);n<t;n++)r[n]=e[n];return r}var s=function(e){var t=l.findIndex((function(t){return t.key===e}));return-1===t?999:t},u=function(e){return l.find((function(t){return t.key===e}))||!1},m=function(e){for(var t,n=[],a=i(e.sort((function(e,t){return function(e,t){var n=s(e),r=s(t);return n>r?1:n<r?-1:0}(e,t)})));!(t=a()).done;){var o=t.value,l="bx--tag";u(o)?l+=" bx--tag--green":l+=" bx--tag--cool-gray",n.push(r.createElement("button",{class:l}," ",r.createElement("span",{class:"bx--tag__label",title:u(o).name},o)," "))}return n};t.Z=function(e){return r.createElement(a.StaticQuery,{query:"3151510810",render:function(t){return function(e,t){var n=function(e,t){var n=t.replace("/toolkit/","");return e.allMdx.edges.filter((function(e){return e.node.slug===n}))[0].node}(e,t),l=n.frontmatter,i="/toolkit/"+n.slug,c=r.createElement("div",null,r.createElement("div",{className:o.pb},r.createElement("h4",null,l.title),r.createElement("p",{className:o.pU},l.description)),r.createElement("p",{className:o.pt},m(l.tags)));return r.createElement(a.Link,{to:i,className:o.Gg},c)}(t,e.to)}})}},9195:function(e,t,n){var r=n(7294),a=n(6258);t.Z=function(e){return r.createElement("div",{className:a.fU},e.children)}},6179:function(e,t,n){var r=n(7294),a=n(5414),o=n(5444);function l(e){var t=e.description,n=e.lang,l=e.meta,i=e.image,c=e.title,s=e.pathname,u=(0,o.useStaticQuery)("151170173").site,m=t||u.siteMetadata.description,d=null;i&&(d=i.includes("https://")||i.includes("http://")?i:u.siteMetadata.siteUrl+"/"+i);var f=s?""+u.siteMetadata.siteUrl+s:null;return r.createElement(a.q,{htmlAttributes:{lang:n},title:c,titleTemplate:"%s | "+u.siteMetadata.title,link:f?[{rel:"canonical",href:f}]:[],meta:[{name:"description",content:m},{property:"og:title",content:c},{property:"og:description",content:m},{property:"og:type",content:"website"},{name:"twitter:title",content:c},{name:"twitter:description",content:m}].concat(i?[{property:"og:image",content:d},{property:"og:image:width",content:i.width},{property:"og:image:height",content:i.height},{name:"twitter:card",content:"summary_large_image"}]:[{name:"twitter:card",content:"summary"}]).concat(l)})}l.defaultProps={lang:"en",meta:[],description:""},t.Z=l},9214:function(e,t,n){n.d(t,{Z:function(){return _}});var r=n(7294),a=n(8650),o=n.n(a),l=n(5444),i=n(4983),c=n(5426),s=n(8477),u=n(808),m=n(8318),d=n(4275),f=n(9851),p=n(2881),g=n(6958),h=n(6156),b=n(2565);function v(e,t){var n="undefined"!=typeof Symbol&&e[Symbol.iterator]||e["@@iterator"];if(n)return(n=n.call(e)).next.bind(n);if(Array.isArray(e)||(n=function(e,t){if(!e)return;if("string"==typeof e)return y(e,t);var n=Object.prototype.toString.call(e).slice(8,-1);"Object"===n&&e.constructor&&(n=e.constructor.name);if("Map"===n||"Set"===n)return Array.from(e);if("Arguments"===n||/^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n))return y(e,t)}(e))||t&&e&&"number"==typeof e.length){n&&(e=n);var r=0;return function(){return r>=e.length?{done:!0}:{done:!1,value:e[r++]}}}throw new TypeError("Invalid attempt to iterate non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.")}function y(e,t){(null==t||t>e.length)&&(t=e.length);for(var n=0,r=new Array(t);n<t;n++)r[n]=e[n];return r}var E=function(e){var t=b.findIndex((function(t){return t.key===e}));return-1===t?999:t},k=function(e){return b.find((function(t){return t.key===e}))||!1},w=function(e){for(var t,n=e.frontmatter,a=n.url,o=[],i=v(n.tags.sort((function(e,t){return function(e,t){var n=E(e),r=E(t);return n>r?1:n<r?-1:0}(e,t)})));!(t=i()).done;){var c=t.value,s="bx--tag";k(c)?s+=" bx--tag--green":s+=" bx--tag--cool-gray",o.push(r.createElement(l.Link,{to:"/toolkit#"+c,key:c},r.createElement("button",{class:s}," ",r.createElement("span",{class:"bx--tag__label",title:k(c).name},c)," ")))}return r.createElement("div",{className:"bx--grid"},r.createElement("div",{className:"bx--row"},r.createElement("div",{className:"bx--col-lg-1"},"Repository: "),r.createElement("div",{className:"bx--col-lg-4"},r.createElement("a",{href:a,target:"_blank",rel:"noreferrer"},a))),r.createElement("div",{className:"bx--row"},r.createElement("div",{className:"bx--col-lg-1 category-header"},"Categories:"),r.createElement("div",{className:"bx--col-lg-4"},r.createElement("div",{className:"RepoHeader-module--flex_sm--FX8Eh"},o))))},N=n(7574),x=n(9195),Z=n(1517),S=n(3167),M=n(1109),P=n(6179),_=function(e){var t=e.pageContext,n=e.children,a=e.location,b=e.Title,v=t.frontmatter,y=void 0===v?{}:v,E=t.relativePagePath,k=t.titleType,_=y.tabs,A=y.title,L=y.theme,T=y.description,C=y.keywords,I=y.date,D=(0,g.Z)().interiorTheme,U={RepoLink:N.Z,RepoLinkList:x.Z,Link:l.Link,ButtonLink:Z.Z,Person:S.Z,People:M.Z,Seo:P.Z},j=(0,l.useStaticQuery)("2102389209").site.pathPrefix,O=j?a.pathname.replace(j,""):a.pathname,H=_?O.split("/").filter(Boolean).slice(-1)[0]||o()(_[0],{lower:!0}):"",B=L||D;return r.createElement(s.Z,{tabs:_,homepage:!1,theme:B,pageTitle:A,pageDescription:T,pageKeywords:C,titleType:k},r.createElement(u.Z,{title:b?r.createElement(b,null):A,label:"label",tabs:_,theme:B}),_&&r.createElement(f.Z,{title:A,slug:O,tabs:_,currentTab:H}),r.createElement(p.Z,{padded:!0},r.createElement(w,{frontmatter:y}),r.createElement(i.Zo,{components:U},n),r.createElement(m.Z,{relativePagePath:E}),r.createElement(h.Z,{date:I})),r.createElement(d.Z,{pageContext:t,location:a,slug:O,tabs:_,currentTab:H}),r.createElement(c.Z,null))}}}]);
//# sourceMappingURL=component---src-pages-toolkit-tm-gcn-mdx-6649622b62940ddcafcd.js.map