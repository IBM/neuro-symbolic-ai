"use strict";(self.webpackChunkneuro_symbolic_ai_toolkit_site=self.webpackChunkneuro_symbolic_ai_toolkit_site||[]).push([[6131],{8337:function(e,t,n){n.r(t),n.d(t,{_frontmatter:function(){return i},default:function(){return s}});var r=n(3366),a=(n(7294),n(4983)),o=n(874),l=["components"],i={},c={_frontmatter:i},u=o.Z;function s(e){var t=e.components,n=(0,r.Z)(e,l);return(0,a.kt)(u,Object.assign({},c,n,{components:t,mdxType:"MDXLayout"}),(0,a.kt)("h2",null,"Description"),(0,a.kt)("p",null,"GPU scaling of Propositional LNN for Word Sense Disambiguation and Text Word Common Sense - implementation using sparse-tensors with bounds, downward and an AND/NOT representation"),(0,a.kt)("h2",null,"Main Contributors"),(0,a.kt)("p",null,"Anamitra Roy Choudhury, Venkatesan Chakaravarthy, Ananda Pal, Yogish Sabharwal"))}s.isMDXComponent=!0},6156:function(e,t,n){n.d(t,{Z:function(){return o}});var r=n(7294),a=n(36),o=function(e){var t=e.date,n=new Date(t);return t?r.createElement(a.X2,{className:"last-modified-date-module--row--XJoYQ"},r.createElement(a.sg,null,r.createElement("div",{className:"last-modified-date-module--text--ogPQF"},"Page last updated: ",n.toLocaleDateString("en-GB",{day:"2-digit",year:"numeric",month:"long"})))):null}},7574:function(e,t,n){var r=n(7294),a=n(5444),o=n(6258),l=n(2565);function i(e,t){var n="undefined"!=typeof Symbol&&e[Symbol.iterator]||e["@@iterator"];if(n)return(n=n.call(e)).next.bind(n);if(Array.isArray(e)||(n=function(e,t){if(!e)return;if("string"==typeof e)return c(e,t);var n=Object.prototype.toString.call(e).slice(8,-1);"Object"===n&&e.constructor&&(n=e.constructor.name);if("Map"===n||"Set"===n)return Array.from(e);if("Arguments"===n||/^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n))return c(e,t)}(e))||t&&e&&"number"==typeof e.length){n&&(e=n);var r=0;return function(){return r>=e.length?{done:!0}:{done:!1,value:e[r++]}}}throw new TypeError("Invalid attempt to iterate non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.")}function c(e,t){(null==t||t>e.length)&&(t=e.length);for(var n=0,r=new Array(t);n<t;n++)r[n]=e[n];return r}var u=function(e){var t=l.findIndex((function(t){return t.key===e}));return-1===t?999:t},s=function(e){return l.find((function(t){return t.key===e}))||!1},m=function(e){for(var t,n=[],a=i(e.sort((function(e,t){return function(e,t){var n=u(e),r=u(t);return n>r?1:n<r?-1:0}(e,t)})));!(t=a()).done;){var o=t.value,l="bx--tag";s(o)?l+=" bx--tag--green":l+=" bx--tag--cool-gray",n.push(r.createElement("button",{class:l}," ",r.createElement("span",{class:"bx--tag__label",title:s(o).name},o)," "))}return n};t.Z=function(e){return r.createElement(a.StaticQuery,{query:"3281138953",render:function(t){return function(e,t){var n=function(e,t){var n=t.replace("/repos/","");return e.allMdx.edges.filter((function(e){return e.node.slug===n}))[0].node}(e,t),l=n.frontmatter,i="/repos/"+n.slug,c=r.createElement("div",null,r.createElement("div",{className:o.pb},r.createElement("h4",null,l.title),r.createElement("p",{className:o.pU},l.description)),r.createElement("p",{className:o.pt},m(l.tags)));return r.createElement(a.Link,{to:i,className:o.Gg},c)}(t,e.to)}})}},9195:function(e,t,n){var r=n(7294),a=n(6258);t.Z=function(e){return r.createElement("div",{className:a.fU},e.children)}},874:function(e,t,n){n.d(t,{Z:function(){return Z}});var r=n(7294),a=n(8650),o=n.n(a),l=n(5444),i=n(4983),c=n(5426),u=n(4311),s=n(808),m=n(8318),f=n(4275),d=n(9851),p=n(2881),b=n(6958),g=n(6156),y=n(2565);function v(e,t){var n="undefined"!=typeof Symbol&&e[Symbol.iterator]||e["@@iterator"];if(n)return(n=n.call(e)).next.bind(n);if(Array.isArray(e)||(n=function(e,t){if(!e)return;if("string"==typeof e)return h(e,t);var n=Object.prototype.toString.call(e).slice(8,-1);"Object"===n&&e.constructor&&(n=e.constructor.name);if("Map"===n||"Set"===n)return Array.from(e);if("Arguments"===n||/^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n))return h(e,t)}(e))||t&&e&&"number"==typeof e.length){n&&(e=n);var r=0;return function(){return r>=e.length?{done:!0}:{done:!1,value:e[r++]}}}throw new TypeError("Invalid attempt to iterate non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.")}function h(e,t){(null==t||t>e.length)&&(t=e.length);for(var n=0,r=new Array(t);n<t;n++)r[n]=e[n];return r}var E=function(e){var t=y.findIndex((function(t){return t.key===e}));return-1===t?999:t},x=function(e){return y.find((function(t){return t.key===e}))||!1},k=function(e){for(var t,n=e.frontmatter,a=n.url,o=[],i=v(n.tags.sort((function(e,t){return function(e,t){var n=E(e),r=E(t);return n>r?1:n<r?-1:0}(e,t)})));!(t=i()).done;){var c=t.value,u="bx--tag";x(c)?u+=" bx--tag--green":u+=" bx--tag--cool-gray",o.push(r.createElement(l.Link,{to:"/repos#"+c,key:c},r.createElement("button",{class:u}," ",r.createElement("span",{class:"bx--tag__label",title:x(c).name},c)," ")))}return r.createElement("div",{className:"bx--grid"},r.createElement("div",{className:"bx--row"},r.createElement("div",{className:"bx--col-lg-1"},"Repository: "),r.createElement("div",{className:"bx--col-lg-4"},r.createElement("a",{href:a,target:"_blank",rel:"noreferrer"},a))),r.createElement("div",{className:"bx--row"},r.createElement("div",{className:"bx--col-lg-1 category-header"},"Categories:"),r.createElement("div",{className:"bx--col-lg-4"},r.createElement("div",{className:"RepoHeader-module--flex_sm--FX8Eh"},o))))},w=n(7574),N=n(9195),Z=function(e){var t=e.pageContext,n=e.children,a=e.location,y=e.Title,v=t.frontmatter,h=void 0===v?{}:v,E=t.relativePagePath,x=t.titleType,Z=h.tabs,S=h.title,_=h.theme,A=h.description,C=h.keywords,T=h.date,L=(0,b.Z)().interiorTheme,P={RepoLink:w.Z,RepoLinkList:N.Z,Link:l.Link},D=(0,l.useStaticQuery)("2102389209").site.pathPrefix,I=D?a.pathname.replace(D,""):a.pathname,j=Z?I.split("/").filter(Boolean).slice(-1)[0]||o()(Z[0],{lower:!0}):"",M=_||L;return r.createElement(u.Z,{tabs:Z,homepage:!1,theme:M,pageTitle:S,pageDescription:A,pageKeywords:C,titleType:x},r.createElement(s.Z,{title:y?r.createElement(y,null):S,label:"label",tabs:Z,theme:M}),Z&&r.createElement(d.Z,{title:S,slug:I,tabs:Z,currentTab:j}),r.createElement(p.Z,{padded:!0},r.createElement(k,{frontmatter:h}),r.createElement(i.Zo,{components:P},n),r.createElement(m.Z,{relativePagePath:E}),r.createElement(g.Z,{date:T})),r.createElement(f.Z,{pageContext:t,location:a,slug:I,tabs:Z,currentTab:j}),r.createElement(c.Z,null))}}}]);
//# sourceMappingURL=component---src-pages-repos-tensorlnn-mdx-ce9673684c9773f55fe5.js.map