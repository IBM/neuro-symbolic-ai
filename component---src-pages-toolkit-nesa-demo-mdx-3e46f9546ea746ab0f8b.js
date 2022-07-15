"use strict";(self.webpackChunkneuro_symbolic_ai_toolkit_site=self.webpackChunkneuro_symbolic_ai_toolkit_site||[]).push([[6176],{735:function(e,t,n){n.r(t),n.d(t,{_frontmatter:function(){return i},default:function(){return s}});var r=n(3366),a=(n(7294),n(4983)),o=n(9214),l=["components"],i={},c={_frontmatter:i},u=o.Z;function s(e){var t=e.components,n=(0,r.Z)(e,l);return(0,a.kt)(u,Object.assign({},c,n,{components:t,mdxType:"MDXLayout"}),(0,a.kt)("h2",null,"Description"),(0,a.kt)("p",null,"This is the HCI component of NeSA. It allows the user to visualize the logical facts, learned policy, accuracy and other metrics. In the future, this will also allow the user to edit the knowledge and the learned policy. It also supports a general purpose visualization and editing tool for any LNN based network."),(0,a.kt)("h2",null,"Main Contributors"),(0,a.kt)("p",null,"Daiki Kimura, Steve Carrow, Stefan Zecevic"))}s.isMDXComponent=!0},6156:function(e,t,n){n.d(t,{Z:function(){return o}});var r=n(7294),a=n(36),o=function(e){var t=e.date,n=new Date(t);return t?r.createElement(a.X2,{className:"last-modified-date-module--row--XJoYQ"},r.createElement(a.sg,null,r.createElement("div",{className:"last-modified-date-module--text--ogPQF"},"Page last updated: ",n.toLocaleDateString("en-GB",{day:"2-digit",year:"numeric",month:"long"})))):null}},1517:function(e,t,n){var r=n(7294);t.Z=function(e){return r.createElement("a",{href:e.href,target:"_blank",rel:"noreferrer",title:e.title,className:"bx--btn bx--btn--primary"},e.children)}},1109:function(e,t,n){n.d(t,{Z:function(){return a}});var r=n(7294),a=function(e){return r.createElement("div",{className:"People-module--flex--LApLg"},e.children)}},3167:function(e,t,n){n.d(t,{Z:function(){return a}});var r=n(7294),a=function(e){var t=e.name,n=e.url,a=e.affiliation,o=e.img;return r.createElement("a",{href:n,target:"_blank",rel:"noreferrer",className:"Person-module--link---uaJL"},r.createElement("div",null,r.createElement("img",{src:o,alt:t,title:t,className:"Person-module--roundImg--l8Swm"}),r.createElement("div",{className:"Person-module--textReset--Y0VEx"},r.createElement("div",{className:"Person-module--name--xrah4"},t),r.createElement("div",{className:"Person-module--affiliation--t+v4k"},a))))}},7574:function(e,t,n){var r=n(7294),a=n(5444),o=n(6258),l=n(2565);function i(e,t){var n="undefined"!=typeof Symbol&&e[Symbol.iterator]||e["@@iterator"];if(n)return(n=n.call(e)).next.bind(n);if(Array.isArray(e)||(n=function(e,t){if(!e)return;if("string"==typeof e)return c(e,t);var n=Object.prototype.toString.call(e).slice(8,-1);"Object"===n&&e.constructor&&(n=e.constructor.name);if("Map"===n||"Set"===n)return Array.from(e);if("Arguments"===n||/^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n))return c(e,t)}(e))||t&&e&&"number"==typeof e.length){n&&(e=n);var r=0;return function(){return r>=e.length?{done:!0}:{done:!1,value:e[r++]}}}throw new TypeError("Invalid attempt to iterate non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.")}function c(e,t){(null==t||t>e.length)&&(t=e.length);for(var n=0,r=new Array(t);n<t;n++)r[n]=e[n];return r}var u=function(e){var t=l.findIndex((function(t){return t.key===e}));return-1===t?999:t},s=function(e){return l.find((function(t){return t.key===e}))||!1},m=function(e){for(var t,n=[],a=i(e.sort((function(e,t){return function(e,t){var n=u(e),r=u(t);return n>r?1:n<r?-1:0}(e,t)})));!(t=a()).done;){var o=t.value,l="bx--tag";s(o)?l+=" bx--tag--green":l+=" bx--tag--cool-gray",n.push(r.createElement("button",{class:l}," ",r.createElement("span",{class:"bx--tag__label",title:s(o).name},o)," "))}return n};t.Z=function(e){return r.createElement(a.StaticQuery,{query:"3151510810",render:function(t){return function(e,t){var n=function(e,t){var n=t.replace("/toolkit/","");return e.allMdx.edges.filter((function(e){return e.node.slug===n}))[0].node}(e,t),l=n.frontmatter,i="/toolkit/"+n.slug,c=r.createElement("div",null,r.createElement("div",{className:o.pb},r.createElement("h4",null,l.title),r.createElement("p",{className:o.pU},l.description)),r.createElement("p",{className:o.pt},m(l.tags)));return r.createElement(a.Link,{to:i,className:o.Gg},c)}(t,e.to)}})}},9195:function(e,t,n){var r=n(7294),a=n(6258);t.Z=function(e){return r.createElement("div",{className:a.fU},e.children)}},6179:function(e,t,n){var r=n(7294),a=n(5414),o=n(5444);function l(e){var t=e.description,n=e.lang,l=e.meta,i=e.image,c=e.title,u=e.pathname,s=(0,o.useStaticQuery)("151170173").site,m=t||s.siteMetadata.description,d=null;i&&(d=i.includes("https://")||i.includes("http://")?i:s.siteMetadata.siteUrl+"/"+i);var f=u?""+s.siteMetadata.siteUrl+u:null;return r.createElement(a.q,{htmlAttributes:{lang:n},title:c,titleTemplate:"%s | "+s.siteMetadata.title,link:f?[{rel:"canonical",href:f}]:[],meta:[{name:"description",content:m},{property:"og:title",content:c},{property:"og:description",content:m},{property:"og:type",content:"website"},{name:"twitter:title",content:c},{name:"twitter:description",content:m}].concat(i?[{property:"og:image",content:d},{property:"og:image:width",content:i.width},{property:"og:image:height",content:i.height},{name:"twitter:card",content:"summary_large_image"}]:[{name:"twitter:card",content:"summary"}]).concat(l)})}l.defaultProps={lang:"en",meta:[],description:""},t.Z=l},9214:function(e,t,n){n.d(t,{Z:function(){return A}});var r=n(7294),a=n(8650),o=n.n(a),l=n(5444),i=n(4983),c=n(5426),u=n(8477),s=n(808),m=n(8318),d=n(4275),f=n(9851),p=n(2881),g=n(6958),v=n(6156),b=n(2565);function h(e,t){var n="undefined"!=typeof Symbol&&e[Symbol.iterator]||e["@@iterator"];if(n)return(n=n.call(e)).next.bind(n);if(Array.isArray(e)||(n=function(e,t){if(!e)return;if("string"==typeof e)return y(e,t);var n=Object.prototype.toString.call(e).slice(8,-1);"Object"===n&&e.constructor&&(n=e.constructor.name);if("Map"===n||"Set"===n)return Array.from(e);if("Arguments"===n||/^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n))return y(e,t)}(e))||t&&e&&"number"==typeof e.length){n&&(e=n);var r=0;return function(){return r>=e.length?{done:!0}:{done:!1,value:e[r++]}}}throw new TypeError("Invalid attempt to iterate non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.")}function y(e,t){(null==t||t>e.length)&&(t=e.length);for(var n=0,r=new Array(t);n<t;n++)r[n]=e[n];return r}var E=function(e){var t=b.findIndex((function(t){return t.key===e}));return-1===t?999:t},k=function(e){return b.find((function(t){return t.key===e}))||!1},x=function(e){for(var t,n=e.frontmatter,a=n.url,o=[],i=h(n.tags.sort((function(e,t){return function(e,t){var n=E(e),r=E(t);return n>r?1:n<r?-1:0}(e,t)})));!(t=i()).done;){var c=t.value,u="bx--tag";k(c)?u+=" bx--tag--green":u+=" bx--tag--cool-gray",o.push(r.createElement(l.Link,{to:"/toolkit#"+c,key:c},r.createElement("button",{class:u}," ",r.createElement("span",{class:"bx--tag__label",title:k(c).name},c)," ")))}return r.createElement("div",{className:"bx--grid"},r.createElement("div",{className:"bx--row"},r.createElement("div",{className:"bx--col-lg-1"},"Repository: "),r.createElement("div",{className:"bx--col-lg-4"},r.createElement("a",{href:a,target:"_blank",rel:"noreferrer"},a))),r.createElement("div",{className:"bx--row"},r.createElement("div",{className:"bx--col-lg-1 category-header"},"Categories:"),r.createElement("div",{className:"bx--col-lg-4"},r.createElement("div",{className:"RepoHeader-module--flex_sm--FX8Eh"},o))))},w=n(7574),Z=n(9195),N=n(1517),_=n(3167),S=n(1109),P=n(6179),A=function(e){var t=e.pageContext,n=e.children,a=e.location,b=e.Title,h=t.frontmatter,y=void 0===h?{}:h,E=t.relativePagePath,k=t.titleType,A=y.tabs,L=y.title,I=y.theme,T=y.description,C=y.keywords,M=y.date,j=(0,g.Z)().interiorTheme,D={RepoLink:w.Z,RepoLinkList:Z.Z,Link:l.Link,ButtonLink:N.Z,Person:_.Z,People:S.Z,Seo:P.Z},U=(0,l.useStaticQuery)("2102389209").site.pathPrefix,O=U?a.pathname.replace(U,""):a.pathname,Q=A?O.split("/").filter(Boolean).slice(-1)[0]||o()(A[0],{lower:!0}):"",R=I||j;return r.createElement(u.Z,{tabs:A,homepage:!1,theme:R,pageTitle:L,pageDescription:T,pageKeywords:C,titleType:k},r.createElement(s.Z,{title:b?r.createElement(b,null):L,label:"label",tabs:A,theme:R}),A&&r.createElement(f.Z,{title:L,slug:O,tabs:A,currentTab:Q}),r.createElement(p.Z,{padded:!0},r.createElement(x,{frontmatter:y}),r.createElement(i.Zo,{components:D},n),r.createElement(m.Z,{relativePagePath:E}),r.createElement(v.Z,{date:M})),r.createElement(d.Z,{pageContext:t,location:a,slug:O,tabs:A,currentTab:Q}),r.createElement(c.Z,null))}}}]);
//# sourceMappingURL=component---src-pages-toolkit-nesa-demo-mdx-3e46f9546ea746ab0f8b.js.map