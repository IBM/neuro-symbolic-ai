"use strict";(self.webpackChunkneuro_symbolic_ai_toolkit_site=self.webpackChunkneuro_symbolic_ai_toolkit_site||[]).push([[2130],{3938:function(e,t,n){n.r(t),n.d(t,{_frontmatter:function(){return l},default:function(){return u}});var r=n(3366),a=(n(7294),n(4983)),o=n(9214),i=["components"],l={},c={_frontmatter:l},s=o.Z;function u(e){var t=e.components,n=(0,r.Z)(e,i);return(0,a.kt)(s,Object.assign({},c,n,{components:t,mdxType:"MDXLayout"}),(0,a.kt)("h2",null,"Description"),(0,a.kt)("p",null,"Natural Language Inference is fundamental to many Natural Language Processing applications such as semantic search and question answering. The task of NLI has gained significant attention in the recent times due to the release of fairly large scale, challenging datasets. Present approaches that address NLI are largely focused on learning based on the given text in order to classify whether the given premise entails, contradicts, or is neutral to the given hypothesis. On the other hand, techniques for Inference, as a central topic in artificial intelligence, has had knowledge bases playing an important role, in particular for formal reasoning tasks. While, there are many open knowledge bases that comprise of various types of information, their use for natural language inference has not been well explored. In this work, we present a simple technique that can harnesses knowledge bases, provided in the form of a graph, for natural language inference."),(0,a.kt)("h2",null,"Main Contributors"),(0,a.kt)("p",null,"Pavan Kapanipathi"))}u.isMDXComponent=!0},6156:function(e,t,n){n.d(t,{Z:function(){return o}});var r=n(7294),a=n(36),o=function(e){var t=e.date,n=new Date(t);return t?r.createElement(a.X2,{className:"last-modified-date-module--row--XJoYQ"},r.createElement(a.sg,null,r.createElement("div",{className:"last-modified-date-module--text--ogPQF"},"Page last updated: ",n.toLocaleDateString("en-GB",{day:"2-digit",year:"numeric",month:"long"})))):null}},1517:function(e,t,n){var r=n(7294);t.Z=function(e){return r.createElement("a",{href:e.href,target:"_blank",rel:"noreferrer",title:e.title,className:"bx--btn bx--btn--primary"},e.children)}},1109:function(e,t,n){n.d(t,{Z:function(){return a}});var r=n(7294),a=function(e){return r.createElement("div",{className:"People-module--flex--LApLg"},e.children)}},3167:function(e,t,n){n.d(t,{Z:function(){return a}});var r=n(7294),a=function(e){var t=e.name,n=e.url,a=e.affiliation,o=e.img;return r.createElement("a",{href:n,target:"_blank",rel:"noreferrer",className:"Person-module--link---uaJL"},r.createElement("div",null,r.createElement("img",{src:o,alt:t,title:t,className:"Person-module--roundImg--l8Swm"}),r.createElement("div",{className:"Person-module--textReset--Y0VEx"},r.createElement("div",{className:"Person-module--name--xrah4"},t),r.createElement("div",{className:"Person-module--affiliation--t+v4k"},a))))}},7574:function(e,t,n){var r=n(7294),a=n(5444),o=n(6258),i=n(2565);function l(e,t){var n="undefined"!=typeof Symbol&&e[Symbol.iterator]||e["@@iterator"];if(n)return(n=n.call(e)).next.bind(n);if(Array.isArray(e)||(n=function(e,t){if(!e)return;if("string"==typeof e)return c(e,t);var n=Object.prototype.toString.call(e).slice(8,-1);"Object"===n&&e.constructor&&(n=e.constructor.name);if("Map"===n||"Set"===n)return Array.from(e);if("Arguments"===n||/^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n))return c(e,t)}(e))||t&&e&&"number"==typeof e.length){n&&(e=n);var r=0;return function(){return r>=e.length?{done:!0}:{done:!1,value:e[r++]}}}throw new TypeError("Invalid attempt to iterate non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.")}function c(e,t){(null==t||t>e.length)&&(t=e.length);for(var n=0,r=new Array(t);n<t;n++)r[n]=e[n];return r}var s=function(e){var t=i.findIndex((function(t){return t.key===e}));return-1===t?999:t},u=function(e){return i.find((function(t){return t.key===e}))||!1},m=function(e){for(var t,n=[],a=l(e.sort((function(e,t){return function(e,t){var n=s(e),r=s(t);return n>r?1:n<r?-1:0}(e,t)})));!(t=a()).done;){var o=t.value,i="bx--tag";u(o)?i+=" bx--tag--green":i+=" bx--tag--cool-gray",n.push(r.createElement("button",{class:i}," ",r.createElement("span",{class:"bx--tag__label",title:u(o).name},o)," "))}return n};t.Z=function(e){return r.createElement(a.StaticQuery,{query:"3151510810",render:function(t){return function(e,t){var n=function(e,t){var n=t.replace("/toolkit/","");return e.allMdx.edges.filter((function(e){return e.node.slug===n}))[0].node}(e,t),i=n.frontmatter,l="/toolkit/"+n.slug,c=r.createElement("div",null,r.createElement("div",{className:o.pb},r.createElement("h4",null,i.title),r.createElement("p",{className:o.pU},i.description)),r.createElement("p",{className:o.pt},m(i.tags)));return r.createElement(a.Link,{to:l,className:o.Gg},c)}(t,e.to)}})}},9195:function(e,t,n){var r=n(7294),a=n(6258);t.Z=function(e){return r.createElement("div",{className:a.fU},e.children)}},6179:function(e,t,n){var r=n(7294),a=n(5414),o=n(5444);function i(e){var t=e.description,n=e.lang,i=e.meta,l=e.image,c=e.title,s=e.pathname,u=(0,o.useStaticQuery)("151170173").site,m=t||u.siteMetadata.description,f=null;l&&(f=l.includes("https://")||l.includes("http://")?l:u.siteMetadata.siteUrl+"/"+l);var d=s?""+u.siteMetadata.siteUrl+s:null;return r.createElement(a.q,{htmlAttributes:{lang:n},title:c,titleTemplate:"%s | "+u.siteMetadata.title,link:d?[{rel:"canonical",href:d}]:[],meta:[{name:"description",content:m},{property:"og:title",content:c},{property:"og:description",content:m},{property:"og:type",content:"website"},{name:"twitter:title",content:c},{name:"twitter:description",content:m}].concat(l?[{property:"og:image",content:f},{property:"og:image:width",content:l.width},{property:"og:image:height",content:l.height},{name:"twitter:card",content:"summary_large_image"}]:[{name:"twitter:card",content:"summary"}]).concat(i)})}i.defaultProps={lang:"en",meta:[],description:""},t.Z=i},9214:function(e,t,n){n.d(t,{Z:function(){return S}});var r=n(7294),a=n(8650),o=n.n(a),i=n(5444),l=n(4983),c=n(5426),s=n(8477),u=n(808),m=n(8318),f=n(4275),d=n(9851),p=n(2881),g=n(6958),h=n(6156),b=n(2565);function v(e,t){var n="undefined"!=typeof Symbol&&e[Symbol.iterator]||e["@@iterator"];if(n)return(n=n.call(e)).next.bind(n);if(Array.isArray(e)||(n=function(e,t){if(!e)return;if("string"==typeof e)return y(e,t);var n=Object.prototype.toString.call(e).slice(8,-1);"Object"===n&&e.constructor&&(n=e.constructor.name);if("Map"===n||"Set"===n)return Array.from(e);if("Arguments"===n||/^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n))return y(e,t)}(e))||t&&e&&"number"==typeof e.length){n&&(e=n);var r=0;return function(){return r>=e.length?{done:!0}:{done:!1,value:e[r++]}}}throw new TypeError("Invalid attempt to iterate non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.")}function y(e,t){(null==t||t>e.length)&&(t=e.length);for(var n=0,r=new Array(t);n<t;n++)r[n]=e[n];return r}var E=function(e){var t=b.findIndex((function(t){return t.key===e}));return-1===t?999:t},k=function(e){return b.find((function(t){return t.key===e}))||!1},x=function(e){for(var t,n=e.frontmatter,a=n.url,o=[],l=v(n.tags.sort((function(e,t){return function(e,t){var n=E(e),r=E(t);return n>r?1:n<r?-1:0}(e,t)})));!(t=l()).done;){var c=t.value,s="bx--tag";k(c)?s+=" bx--tag--green":s+=" bx--tag--cool-gray",o.push(r.createElement(i.Link,{to:"/toolkit#"+c,key:c},r.createElement("button",{class:s}," ",r.createElement("span",{class:"bx--tag__label",title:k(c).name},c)," ")))}return r.createElement("div",{className:"bx--grid"},r.createElement("div",{className:"bx--row"},r.createElement("div",{className:"bx--col-lg-1"},"Repository: "),r.createElement("div",{className:"bx--col-lg-4"},r.createElement("a",{href:a,target:"_blank",rel:"noreferrer"},a))),r.createElement("div",{className:"bx--row"},r.createElement("div",{className:"bx--col-lg-1 category-header"},"Categories:"),r.createElement("div",{className:"bx--col-lg-4"},r.createElement("div",{className:"RepoHeader-module--flex_sm--FX8Eh"},o))))},w=n(7574),N=n(9195),Z=n(1517),_=n(3167),P=n(1109),L=n(6179),S=function(e){var t=e.pageContext,n=e.children,a=e.location,b=e.Title,v=t.frontmatter,y=void 0===v?{}:v,E=t.relativePagePath,k=t.titleType,S=y.tabs,A=y.title,I=y.theme,T=y.description,M=y.keywords,C=y.date,j=(0,g.Z)().interiorTheme,D={RepoLink:w.Z,RepoLinkList:N.Z,Link:i.Link,ButtonLink:Z.Z,Person:_.Z,People:P.Z,Seo:L.Z},O=(0,i.useStaticQuery)("2102389209").site.pathPrefix,U=O?a.pathname.replace(O,""):a.pathname,q=S?U.split("/").filter(Boolean).slice(-1)[0]||o()(S[0],{lower:!0}):"",Q=I||j;return r.createElement(s.Z,{tabs:S,homepage:!1,theme:Q,pageTitle:A,pageDescription:T,pageKeywords:M,titleType:k},r.createElement(u.Z,{title:b?r.createElement(b,null):A,label:"label",tabs:S,theme:Q}),S&&r.createElement(d.Z,{title:A,slug:U,tabs:S,currentTab:q}),r.createElement(p.Z,{padded:!0},r.createElement(x,{frontmatter:y}),r.createElement(l.Zo,{components:D},n),r.createElement(m.Z,{relativePagePath:E}),r.createElement(h.Z,{date:C})),r.createElement(f.Z,{pageContext:t,location:a,slug:U,tabs:S,currentTab:q}),r.createElement(c.Z,null))}}}]);
//# sourceMappingURL=component---src-pages-toolkit-knowledge-enabled-textual-entailment-mdx-7293e0019128153dd4a5.js.map