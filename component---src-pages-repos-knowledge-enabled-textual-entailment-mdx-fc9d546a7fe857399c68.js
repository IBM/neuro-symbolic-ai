"use strict";(self.webpackChunkneuro_symbolic_ai_toolkit_site=self.webpackChunkneuro_symbolic_ai_toolkit_site||[]).push([[9798],{6932:function(e,t,n){n.r(t),n.d(t,{_frontmatter:function(){return i},default:function(){return u}});var r=n(3366),a=(n(7294),n(4983)),l=n(9985),o=["components"],i={},s={_frontmatter:i},c=l.Z;function u(e){var t=e.components,n=(0,r.Z)(e,o);return(0,a.kt)(c,Object.assign({},s,n,{components:t,mdxType:"MDXLayout"}),(0,a.kt)("h2",null,"Description"),(0,a.kt)("p",null,"Natural Language Inference is fundamental to many Natural Language Processing applications such as semantic search and question answering. The task of NLI has gained significant attention in the recent times due to the release of fairly large scale, challenging datasets. Present approaches that address NLI are largely focused on learning based on the given text in order to classify whether the given premise entails, contradicts, or is neutral to the given hypothesis. On the other hand, techniques for Inference, as a central topic in artificial intelligence, has had knowledge bases playing an important role, in particular for formal reasoning tasks. While, there are many open knowledge bases that comprise of various types of information, their use for natural language inference has not been well explored. In this work, we present a simple technique that can harnesses knowledge bases, provided in the form of a graph, for natural language inference."),(0,a.kt)("h2",null,"Main Contributors"),(0,a.kt)("p",null,"Pavan Kapanipathi"))}u.isMDXComponent=!0},6156:function(e,t,n){n.d(t,{Z:function(){return l}});var r=n(7294),a=n(36),l=function(e){var t=e.date,n=new Date(t);return t?r.createElement(a.X2,{className:"last-modified-date-module--row--XJoYQ"},r.createElement(a.sg,null,r.createElement("div",{className:"last-modified-date-module--text--ogPQF"},"Page last updated: ",n.toLocaleDateString("en-GB",{day:"2-digit",year:"numeric",month:"long"})))):null}},9985:function(e,t,n){n.d(t,{Z:function(){return Z}});var r=n(7294),a=n(8650),l=n.n(a),o=n(5444),i=n(4983),s=n(5426),c=n(4311),u=n(808),m=n(8318),f=n(4275),d=n(9851),p=n(2881),g=n(6958),h=n(6156);function b(e,t){var n="undefined"!=typeof Symbol&&e[Symbol.iterator]||e["@@iterator"];if(n)return(n=n.call(e)).next.bind(n);if(Array.isArray(e)||(n=function(e,t){if(!e)return;if("string"==typeof e)return v(e,t);var n=Object.prototype.toString.call(e).slice(8,-1);"Object"===n&&e.constructor&&(n=e.constructor.name);if("Map"===n||"Set"===n)return Array.from(e);if("Arguments"===n||/^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n))return v(e,t)}(e))||t&&e&&"number"==typeof e.length){n&&(e=n);var r=0;return function(){return r>=e.length?{done:!0}:{done:!1,value:e[r++]}}}throw new TypeError("Invalid attempt to iterate non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.")}function v(e,t){(null==t||t>e.length)&&(t=e.length);for(var n=0,r=new Array(t);n<t;n++)r[n]=e[n];return r}var y=function(e){for(var t,n=e.frontmatter,a=n.url,l=[],i=b(n.tags.entries());!(t=i()).done;){var s=t.value,c=s[0],u=s[1];l.push(r.createElement(o.Link,{to:"/repos#"+u,key:c},r.createElement("button",{class:"bx--tag bx--tag--green"}," ",r.createElement("span",{class:"bx--tag__label"},u)," ")))}return r.createElement("div",{className:"bx--grid"},r.createElement("div",{className:"bx--row"},r.createElement("div",{className:"bx--col-lg-1"},"Repository: "),r.createElement("div",{className:"bx--col-lg-4"},r.createElement("a",{href:a,target:"_blank",rel:"noreferrer"},a))),r.createElement("div",{className:"bx--row"},r.createElement("div",{className:"bx--col-lg-1 category-header"},"Categories:"),r.createElement("div",{className:"bx--col-lg-4"},r.createElement("div",{className:"RepoHeader-module--flex--anwdv"},l))))},E=n(6258);function k(e,t){var n="undefined"!=typeof Symbol&&e[Symbol.iterator]||e["@@iterator"];if(n)return(n=n.call(e)).next.bind(n);if(Array.isArray(e)||(n=function(e,t){if(!e)return;if("string"==typeof e)return w(e,t);var n=Object.prototype.toString.call(e).slice(8,-1);"Object"===n&&e.constructor&&(n=e.constructor.name);if("Map"===n||"Set"===n)return Array.from(e);if("Arguments"===n||/^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n))return w(e,t)}(e))||t&&e&&"number"==typeof e.length){n&&(e=n);var r=0;return function(){return r>=e.length?{done:!0}:{done:!1,value:e[r++]}}}throw new TypeError("Invalid attempt to iterate non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.")}function w(e,t){(null==t||t>e.length)&&(t=e.length);for(var n=0,r=new Array(t);n<t;n++)r[n]=e[n];return r}var x=function(e,t){var n=function(e,t){var n=t.replace("/repos/","");return e.allMdx.edges.filter((function(e){return e.node.slug===n}))[0].node}(e,t),a=n.frontmatter,l="/repos/"+n.slug,i=r.createElement("div",null,r.createElement("div",{className:E.pb},r.createElement("h4",null,a.title),r.createElement("p",{className:E.pU},a.description)),r.createElement("p",{className:E.pt},function(e){for(var t,n=[],a=k(e);!(t=a()).done;){var l=t.value;n.push(r.createElement("button",{class:"bx--tag bx--tag--green"}," ",r.createElement("span",{class:"bx--tag__label"},l)," "))}return n}(a.tags)));return r.createElement(o.Link,{to:l,className:E.Gg},i)},N=function(e){return r.createElement(o.StaticQuery,{query:"3281138953",render:function(t){return x(t,e.to)}})},_=function(e){return r.createElement("div",{className:E.fU},e.children)},Z=function(e){var t=e.pageContext,n=e.children,a=e.location,b=e.Title,v=t.frontmatter,E=void 0===v?{}:v,k=t.relativePagePath,w=t.titleType,x=E.tabs,Z=E.title,S=E.theme,A=E.description,I=E.keywords,L=E.date,T=(0,g.Z)().interiorTheme,P={RepoLink:N,RepoLinkList:_},C=(0,o.useStaticQuery)("2102389209").site.pathPrefix,j=C?a.pathname.replace(C,""):a.pathname,D=x?j.split("/").filter(Boolean).slice(-1)[0]||l()(x[0],{lower:!0}):"",M=S||T;return r.createElement(c.Z,{tabs:x,homepage:!1,theme:M,pageTitle:Z,pageDescription:A,pageKeywords:I,titleType:w},r.createElement(u.Z,{title:b?r.createElement(b,null):Z,label:"label",tabs:x,theme:M}),x&&r.createElement(d.Z,{title:Z,slug:j,tabs:x,currentTab:D}),r.createElement(p.Z,{padded:!0},r.createElement(y,{frontmatter:E}),r.createElement(i.Zo,{components:P},n),r.createElement(m.Z,{relativePagePath:k}),r.createElement(h.Z,{date:L})),r.createElement(f.Z,{pageContext:t,location:a,slug:j,tabs:x,currentTab:D}),r.createElement(s.Z,null))}}}]);
//# sourceMappingURL=component---src-pages-repos-knowledge-enabled-textual-entailment-mdx-fc9d546a7fe857399c68.js.map