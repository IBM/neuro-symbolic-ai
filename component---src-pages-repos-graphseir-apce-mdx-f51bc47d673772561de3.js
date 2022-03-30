"use strict";(self.webpackChunkneuro_symbolic_ai_toolkit_site=self.webpackChunkneuro_symbolic_ai_toolkit_site||[]).push([[9877],{263:function(e,t,r){r.r(t),r.d(t,{_frontmatter:function(){return l},default:function(){return s}});var n=r(3366),a=(r(7294),r(4983)),o=r(874),i=["components"],l={},c={_frontmatter:l},u=o.Z;function s(e){var t=e.components,r=(0,n.Z)(e,i);return(0,a.kt)(u,Object.assign({},c,r,{components:t,mdxType:"MDXLayout"}),(0,a.kt)("h2",null,"Description"),(0,a.kt)("p",null,"Matlab code for Dynamic graph based epidemiological model for COVID-19 contact tracing data analysis and optimal testing prescription. JBI paper (",(0,a.kt)("a",{parentName:"p",href:"https://www.sciencedirect.com/science/article/pii/S1532046421002306"},"https://www.sciencedirect.com/science/article/pii/S1532046421002306"),"), IBM blog (",(0,a.kt)("a",{parentName:"p",href:"https://research.ibm.com/blog/accelerating-covid-discoveries"},"https://research.ibm.com/blog/accelerating-covid-discoveries"),")"),(0,a.kt)("h2",null,"Main Contributors"),(0,a.kt)("p",null,"Shashanka Ubaru, Lior Horesh"))}s.isMDXComponent=!0},6156:function(e,t,r){r.d(t,{Z:function(){return o}});var n=r(7294),a=r(36),o=function(e){var t=e.date,r=new Date(t);return t?n.createElement(a.X2,{className:"last-modified-date-module--row--XJoYQ"},n.createElement(a.sg,null,n.createElement("div",{className:"last-modified-date-module--text--ogPQF"},"Page last updated: ",r.toLocaleDateString("en-GB",{day:"2-digit",year:"numeric",month:"long"})))):null}},7574:function(e,t,r){var n=r(7294),a=r(5444),o=r(6258),i=r(2565);function l(e,t){var r="undefined"!=typeof Symbol&&e[Symbol.iterator]||e["@@iterator"];if(r)return(r=r.call(e)).next.bind(r);if(Array.isArray(e)||(r=function(e,t){if(!e)return;if("string"==typeof e)return c(e,t);var r=Object.prototype.toString.call(e).slice(8,-1);"Object"===r&&e.constructor&&(r=e.constructor.name);if("Map"===r||"Set"===r)return Array.from(e);if("Arguments"===r||/^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(r))return c(e,t)}(e))||t&&e&&"number"==typeof e.length){r&&(e=r);var n=0;return function(){return n>=e.length?{done:!0}:{done:!1,value:e[n++]}}}throw new TypeError("Invalid attempt to iterate non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.")}function c(e,t){(null==t||t>e.length)&&(t=e.length);for(var r=0,n=new Array(t);r<t;r++)n[r]=e[r];return n}var u=function(e){var t=i.findIndex((function(t){return t.key===e}));return-1===t?999:t},s=function(e){return i.find((function(t){return t.key===e}))||!1},m=function(e){for(var t,r=[],a=l(e.sort((function(e,t){return function(e,t){var r=u(e),n=u(t);return r>n?1:r<n?-1:0}(e,t)})));!(t=a()).done;){var o=t.value,i="bx--tag";s(o)?i+=" bx--tag--blue":i+=" bx--tag--green",r.push(n.createElement("button",{class:i}," ",n.createElement("span",{class:"bx--tag__label",title:s(o).name},o)," "))}return r};t.Z=function(e){return n.createElement(a.StaticQuery,{query:"3281138953",render:function(t){return function(e,t){var r=function(e,t){var r=t.replace("/repos/","");return e.allMdx.edges.filter((function(e){return e.node.slug===r}))[0].node}(e,t),i=r.frontmatter,l="/repos/"+r.slug,c=n.createElement("div",null,n.createElement("div",{className:o.pb},n.createElement("h4",null,i.title),n.createElement("p",{className:o.pU},i.description)),n.createElement("p",{className:o.pt},m(i.tags)));return n.createElement(a.Link,{to:l,className:o.Gg},c)}(t,e.to)}})}},9195:function(e,t,r){var n=r(7294),a=r(6258);t.Z=function(e){return n.createElement("div",{className:a.fU},e.children)}},874:function(e,t,r){r.d(t,{Z:function(){return N}});var n=r(7294),a=r(8650),o=r.n(a),i=r(5444),l=r(4983),c=r(5426),u=r(4311),s=r(808),m=r(8318),f=r(4275),d=r(9851),p=r(2881),b=r(6958),g=r(6156),v=r(2565);function h(e,t){var r="undefined"!=typeof Symbol&&e[Symbol.iterator]||e["@@iterator"];if(r)return(r=r.call(e)).next.bind(r);if(Array.isArray(e)||(r=function(e,t){if(!e)return;if("string"==typeof e)return y(e,t);var r=Object.prototype.toString.call(e).slice(8,-1);"Object"===r&&e.constructor&&(r=e.constructor.name);if("Map"===r||"Set"===r)return Array.from(e);if("Arguments"===r||/^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(r))return y(e,t)}(e))||t&&e&&"number"==typeof e.length){r&&(e=r);var n=0;return function(){return n>=e.length?{done:!0}:{done:!1,value:e[n++]}}}throw new TypeError("Invalid attempt to iterate non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.")}function y(e,t){(null==t||t>e.length)&&(t=e.length);for(var r=0,n=new Array(t);r<t;r++)n[r]=e[r];return n}var E=function(e){var t=v.findIndex((function(t){return t.key===e}));return-1===t?999:t},k=function(e){return v.find((function(t){return t.key===e}))||!1},x=function(e){for(var t,r=e.frontmatter,a=r.url,o=[],l=h(r.tags.sort((function(e,t){return function(e,t){var r=E(e),n=E(t);return r>n?1:r<n?-1:0}(e,t)})));!(t=l()).done;){var c=t.value,u="bx--tag";k(c)?u+=" bx--tag--blue":u+=" bx--tag--green",o.push(n.createElement(i.Link,{to:"/repos#"+c,key:c},n.createElement("button",{class:u}," ",n.createElement("span",{class:"bx--tag__label",title:k(c).name},c)," ")))}return n.createElement("div",{className:"bx--grid"},n.createElement("div",{className:"bx--row"},n.createElement("div",{className:"bx--col-lg-1"},"Repository: "),n.createElement("div",{className:"bx--col-lg-4"},n.createElement("a",{href:a,target:"_blank",rel:"noreferrer"},a))),n.createElement("div",{className:"bx--row"},n.createElement("div",{className:"bx--col-lg-1 category-header"},"Categories:"),n.createElement("div",{className:"bx--col-lg-4"},n.createElement("div",{className:"RepoHeader-module--flex_sm--FX8Eh"},o))))},w=r(7574),Z=r(9195),N=function(e){var t=e.pageContext,r=e.children,a=e.location,v=e.Title,h=t.frontmatter,y=void 0===h?{}:h,E=t.relativePagePath,k=t.titleType,N=y.tabs,S=y.title,_=y.theme,A=y.description,I=y.keywords,C=y.date,L=(0,b.Z)().interiorTheme,T={RepoLink:w.Z,RepoLinkList:Z.Z,Link:i.Link},D=(0,i.useStaticQuery)("2102389209").site.pathPrefix,M=D?a.pathname.replace(D,""):a.pathname,j=N?M.split("/").filter(Boolean).slice(-1)[0]||o()(N[0],{lower:!0}):"",P=_||L;return n.createElement(u.Z,{tabs:N,homepage:!1,theme:P,pageTitle:S,pageDescription:A,pageKeywords:I,titleType:k},n.createElement(s.Z,{title:v?n.createElement(v,null):S,label:"label",tabs:N,theme:P}),N&&n.createElement(d.Z,{title:S,slug:M,tabs:N,currentTab:j}),n.createElement(p.Z,{padded:!0},n.createElement(x,{frontmatter:y}),n.createElement(l.Zo,{components:T},r),n.createElement(m.Z,{relativePagePath:E}),n.createElement(g.Z,{date:C})),n.createElement(f.Z,{pageContext:t,location:a,slug:M,tabs:N,currentTab:j}),n.createElement(c.Z,null))}}}]);
//# sourceMappingURL=component---src-pages-repos-graphseir-apce-mdx-f51bc47d673772561de3.js.map